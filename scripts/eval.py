"""Parameter Golf official evaluation script.

Loads a Composer checkpoint, reconstructs the model from the saved architecture
config, and runs the official Parameter Golf BPB evaluation on the FineWeb
validation set using the reference's tokenizer-aware byte counting.

Automatically detects and evaluates the full-precision checkpoint and all
quantized checkpoints found in the checkpoint directory. Supports intN,
mxfp4, and nvfp4 schemes. Results are saved to a single eval_report.json
with per-scheme quant results.

By default, discovers paths and parameters from the current directory:
  - Checkpoint: artifacts/checkpoint/
  - Quantized: artifacts/checkpoint/*_{int*,mxfp*,nvfp*} (auto-discovered)
  - Validation data: derived from train.yaml data path (train -> val)
  - Tokenizer: from model.yaml
  - sequence_length, batch_size: from train.yaml
  - Report: artifacts/eval_report.json

All values are overridable via command-line options.

Usage:
    python eval.py
    python eval.py --checkpoint artifacts/checkpoint
    python eval.py --val-data /path/to/fineweb_val_*.bin
    torchrun --nproc_per_node=8 eval.py
"""

from __future__ import annotations

import glob
import io
import json
import logging
import math
import os
import re
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from composer.distributed import (
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
)
from composer.nn.architecture.model import Model
from composer.train import TrainingRunConfig
from composer.utils.logging import format_float, format_int, setup_logging

logger = logging.getLogger(__name__)


def load_data_shard(file: Path) -> torch.Tensor:
    """Load a single binary token shard with a 1024-byte header."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern: str, seq_len: int) -> torch.Tensor:
    """Load and concatenate all validation shards into a single token tensor."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build lookup tables for tokenizer-aware BPB byte counting.

    The reference Parameter Golf evaluation accounts for SentencePiece-specific
    byte counting: leading space characters (the sentencepiece "▁" marker) are
    only counted when the previous token is not a boundary token.

    Returns:
        Tuple of (base_bytes, has_leading_space, is_boundary_token) tensors.
    """
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def evaluate(
    model: torch.nn.Module,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    seq_len: int,
    batch_seqs: int,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, int]:
    """Run the official Parameter Golf evaluation.

    Computes cross-entropy loss and tokenizer-aware BPB over the full
    validation set using the reference's exact byte counting method.
    Supports multi-GPU via torchrun - each rank processes a strided
    subset of batches, then results are all-reduced.

    Returns:
        Tuple of (val_loss, val_bpb, tokens_evaluated).
    """
    rank = get_rank()
    world_size = get_world_size()
    distributed = is_distributed()

    total_seqs = (val_tokens.numel() - 1) // seq_len
    total_batches = (total_seqs + batch_seqs - 1) // batch_seqs
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    batch_num = 0
    with torch.inference_mode():
        for batch_start in range(0, total_seqs, batch_seqs):
            if max_batches is not None and batch_num >= max_batches:
                break
            # Stripe batches across ranks
            if batch_num % world_size != rank:
                batch_num += 1
                continue
            batch_num += 1
            batch_end = min(batch_start + batch_seqs, total_seqs)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(x, return_dict=False)
                batch_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    y.view(-1),
                    reduction="mean",
                )

            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count

            # Tokenizer-aware byte counting (matches reference exactly)
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(
                dtype=torch.int16
            )
            val_byte_count += token_bytes.to(torch.float64).sum()

            if is_main_process() and (batch_num % 10 == 0 or batch_num == total_batches):
                running_loss = (val_loss_sum / val_token_count).item()
                running_bpt = running_loss / math.log(2.0)
                running_tpb = val_token_count.item() / max(val_byte_count.item(), 1)
                running_bpb = running_bpt * running_tpb
                logger.info(
                    f" - batch {batch_num}/{total_batches}"
                    f" [cyan]loss[/]={format_float(running_loss)}"
                    f" [cyan]bpb[/]={format_float(running_bpb)}"
                )

    # All-reduce accumulators across ranks
    if distributed:
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    return (
        float(val_loss.item()),
        float(bits_per_token * tokens_per_byte),
        int(val_token_count.item()),
    )


def _get_script_dir() -> Path:
    return Path(__file__).resolve().parent


def dequantize_state_dict(obj: dict[str, object]) -> dict[str, torch.Tensor]:
    """Reconstruct a full-precision state dict from a quantized object.

    Imports dequantize_state_dict from quant.py to avoid code duplication
    (handles bit-packing, per-row scales, passthrough, etc.).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("quant", _get_script_dir() / "quant.py")
    quant_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant_mod)
    return quant_mod.dequantize_state_dict(obj)


def _load_train_config(train_yaml: Path) -> dict[str, Any]:
    """Load and resolve the training config, extracting eval-relevant defaults.

    Uses Composer's config loader to fully resolve ``!include``, ``!expr``,
    and ``!env`` tags, then extracts data paths, tokenizer path, batch_size,
    and sequence_length from the resolved config.

    Returns:
        Dict with keys: val_data, tokenizer, batch_size, sequence_length
        (any may be None if not found).
    """
    result: dict[str, Any] = {}
    try:
        config = TrainingRunConfig.model_validate_yaml_file(train_yaml)

        # Extract from the first training stage
        for stage in config.training.values():
            batch_size = getattr(stage, "batch_size", None)
            if batch_size is not None:
                result["batch_size"] = int(batch_size)
            seq_len = getattr(stage, "sequence_length", None)
            if seq_len is not None:
                result["sequence_length"] = int(seq_len)

            # Extract data path and derive val path
            data_config = stage.data
            data_path = getattr(data_config, "path", None)
            if data_path is None:
                # Handle shorthand wrapping
                inner = getattr(data_config, "__pydantic_extra__", {})
                if isinstance(inner, dict):
                    data_path = inner.get("path")
            if isinstance(data_path, str) and "_train_" in data_path:
                result["val_data"] = re.sub(r"_train_", "_val_", data_path)
            break

        # Extract tokenizer path from manifest
        if config.manifest is not None:
            tok_config = config.manifest.get_tokenizer_config(config.model_name)
            if tok_config is not None:
                model_path = getattr(tok_config, "model_path", None)
                if model_path is not None:
                    result["tokenizer"] = model_path
    except Exception:
        pass
    return result


@click.command()
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(),
    help="Path to checkpoint directory (default: artifacts/checkpoint).",
)
@click.option(
    "--schemes",
    default=None,
    type=str,
    help="Quant schemes to evaluate, comma-separated "
    "(default: all discovered). E.g. int6,int8,mxfp4.",
)
@click.option(
    "--step",
    default=None,
    type=int,
    help="Specific checkpoint step to load (latest if omitted).",
)
@click.option(
    "--tokenizer",
    "tokenizer_path",
    default=None,
    type=click.Path(),
    help="Path to SentencePiece .model file (default: from model.yaml).",
)
@click.option(
    "--val-data",
    default=None,
    help="Glob pattern for validation data shards (default: derived from train.yaml).",
)
@click.option(
    "--sequence-length",
    default=None,
    type=int,
    help="Sequence length for evaluation (default: from train.yaml).",
)
@click.option(
    "--batch-size",
    default=None,
    type=int,
    help="Number of sequences per evaluation batch (default: from train.yaml).",
)
@click.option(
    "--max-batches",
    default=None,
    type=int,
    help="Maximum number of batches to evaluate (default: all).",
)
@click.option(
    "--device",
    default=None,
    help="Device to use (default: cuda if available, else cpu).",
)
@click.option(
    "--report",
    "report_path",
    default=None,
    type=click.Path(),
    help="Path to save eval report JSON (default: artifacts/eval_report.json).",
)
def main(
    checkpoint: str | None,
    schemes: str | None,
    step: int | None,
    tokenizer_path: str | None,
    val_data: str | None,
    sequence_length: int | None,
    batch_size: int | None,
    max_batches: int | None,
    device: str | None,
    report_path: str | None,
):
    """Run official Parameter Golf evaluation on a Composer checkpoint."""
    setup_logging()

    # Init distributed if launched via torchrun
    init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Resolve defaults from train.yaml if present
    train_yaml = Path("train.yaml")
    defaults = _load_train_config(train_yaml) if train_yaml.exists() else {}

    # Determine which checkpoints exist
    if checkpoint is None:
        checkpoint = "artifacts/checkpoint"
    checkpoint_dir = Path(checkpoint)
    # Quant scheme suffixes to detect
    _QUANT_SUFFIX_RE = re.compile(r"_(int\d+|mxfp\d+|nvfp\d+)$")

    # Find the non-quantized checkpoint file (resolve to exact path so
    # Model.from_checkpoint doesn't accidentally pick a quant file)
    full_checkpoint_file = None
    if checkpoint_dir.is_dir():
        full_checkpoints = sorted(
            [p for p in checkpoint_dir.glob("step_*") if not _QUANT_SUFFIX_RE.search(p.name)],
            key=lambda p: int(p.name.split("_")[1]),
        )
        if full_checkpoints:
            full_checkpoint_file = full_checkpoints[-1]
    has_full = full_checkpoint_file is not None

    # Auto-discover all quantized files
    quant_files: list[Path] = []
    for search_dir in [checkpoint_dir, checkpoint_dir.parent]:
        if search_dir.is_dir():
            for p in sorted(search_dir.iterdir()):
                if p.is_file() and _QUANT_SUFFIX_RE.search(p.name):
                    quant_files.append(p)
    # Deduplicate by resolved path
    seen = set()
    unique_quant: list[Path] = []
    for qf in quant_files:
        resolved = qf.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_quant.append(qf)
    quant_files = unique_quant

    # Filter to requested schemes if specified
    if schemes is not None:
        requested = {f"_{s.strip()}" for s in schemes.split(",")}
        quant_files = [qf for qf in quant_files if any(qf.name.endswith(r) for r in requested)]

    if not has_full and not quant_files:
        raise click.BadParameter(
            f"No checkpoint found. Looked for:\n"
            f"  Full: {checkpoint}\n"
            f"  Quantized: {checkpoint}/*_<scheme> or {checkpoint_dir.parent}/*_<scheme>"
        )

    if val_data is None:
        val_data = defaults.get("val_data", "data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")

    if tokenizer_path is None:
        tokenizer_path = defaults.get("tokenizer", "data/tokenizers/fineweb_1024_bpe.model")
    if not Path(tokenizer_path).exists():
        raise click.BadParameter(f"Tokenizer not found: {tokenizer_path}")

    if sequence_length is None:
        sequence_length = defaults.get("sequence_length", 1024)
    if batch_size is None:
        batch_size = defaults.get("batch_size", 64)

    if report_path is None:
        report_path = "artifacts/eval_report.json"

    if device is None:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if is_main_process():
        checkpoints_str = []
        if has_full:
            checkpoints_str.append(f"full={checkpoint}")
        for qf in quant_files:
            checkpoints_str.append(f"{qf.name}")
        logger.info(
            f"[bold]Eval config:[/] [cyan]checkpoints[/]={', '.join(checkpoints_str)}"
            f" [cyan]device[/]={device}"
            f" [cyan]sequence_length[/]={sequence_length}"
            f" [cyan]batch_size[/]={batch_size}"
            f" [cyan]world_size[/]={get_world_size()}"
            + (f" [cyan]max_batches[/]={max_batches}" if max_batches is not None else "")
        )
        logger.info(f"[bold]Tokenizer:[/] {tokenizer_path}")
        logger.info(f"[bold]Validation data:[/] {val_data}")

    # Load tokenizer and build BPB lookup tables
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    vocab_size = int(sp.vocab_size())
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, vocab_size, device
    )

    # Load validation data
    val_tokens = load_validation_tokens(val_data, sequence_length)
    total_seqs = (val_tokens.numel() - 1) // sequence_length
    if is_main_process():
        logger.info(
            f"Loaded [cyan]tokens[/]={format_int(val_tokens.numel())}"
            f" [cyan]sequences[/]={format_int(total_seqs)}"
            f" [cyan]vocab_size[/]={vocab_size}"
        )

    # Enable TF32 for eval consistency
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    eval_kwargs = dict(
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        seq_len=sequence_length,
        batch_seqs=batch_size,
        device=device,
        max_batches=max_batches,
    )

    # Load existing report to preserve prior results
    report_file = Path(report_path)
    if report_file.exists():
        with open(report_file) as f:
            report: dict[str, Any] = json.load(f)
    else:
        report: dict[str, Any] = {}

    report.update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "val_tokens_evaluated": val_tokens.numel(),
            "vocab_size": vocab_size,
        }
    )

    # --- Evaluate full-precision model (skip if --schemes filter is set) ---
    base_model = None
    if has_full and schemes is None:
        if is_main_process():
            logger.info("[bold]Loading full-precision checkpoint[/]")
        model = Model.from_checkpoint(full_checkpoint_file, device=device)

        if is_main_process():
            logger.info("[bold]Running full-precision evaluation[/]")
        eval_start = time.perf_counter()
        val_loss, val_bpb, eval_tokens = evaluate(model=model, **eval_kwargs)
        eval_time = time.perf_counter() - eval_start

        if is_main_process():
            logger.info(
                f"[bold green]Full-precision eval complete:[/]"
                f" [cyan]val_loss[/]={format_float(val_loss)}"
                f" [cyan]val_bpb[/]={format_float(val_bpb)}"
                f" in {eval_time:.1f}s"
            )

        report["checkpoint"] = str(checkpoint)
        report["val_loss"] = val_loss
        report["val_bpb"] = val_bpb
        report["val_tokens_evaluated"] = eval_tokens
        report["eval_time_seconds"] = round(eval_time, 2)
        base_model = model

    # --- Evaluate each quantized checkpoint ---
    if quant_files:
        quant_results = report.get("quant", {})

        for qf in quant_files:
            # Extract scheme tag from filename (*_int6 -> int6, *_mxfp4 -> mxfp4)
            match = _QUANT_SUFFIX_RE.search(qf.name)
            if not match:
                continue
            quant_tag = match.group(1)

            if is_main_process():
                logger.info(f"[bold]Loading {quant_tag} checkpoint:[/] {qf.name}")

            with open(qf, "rb") as f:
                quant_blob = f.read()
            # Decompress: try zstd first, fall back to zlib
            try:
                import zstandard

                decompressed = zstandard.ZstdDecompressor().decompress(quant_blob)
            except Exception:
                decompressed = zlib.decompress(quant_blob)
            quant_state = torch.load(
                io.BytesIO(decompressed),
                map_location="cpu",
                weights_only=False,
            )
            dequantized = dequantize_state_dict(quant_state)

            if base_model is not None:
                base_model.load_state_dict(dequantized, strict=True)
                quant_model = base_model
            else:
                # Rebuild model from config embedded in the quantized file
                composer_config = quant_state.get("__composer_config__")
                if composer_config is not None:
                    from composer.nn.architecture.config_types import ArchitectureConfigType
                    from pydantic import TypeAdapter

                    adapter = TypeAdapter(ArchitectureConfigType)
                    arch_config = adapter.validate_python(composer_config)
                    quant_model = arch_config.build(device="cpu")
                else:
                    config = TrainingRunConfig.model_validate_yaml_file(train_yaml)
                    quant_model = config.manifest.get_variant(config.model_name).build(device="cpu")
                quant_model.load_state_dict(dequantized, strict=True)
                base_model = quant_model

            quant_model.to(device)

            if is_main_process():
                logger.info(f"[bold]Running {quant_tag} evaluation[/] (device={device})")
            eval_start = time.perf_counter()
            q_val_loss, q_val_bpb, q_eval_tokens = evaluate(model=quant_model, **eval_kwargs)
            q_eval_time = time.perf_counter() - eval_start

            if is_main_process():
                logger.info(
                    f"[bold green]{quant_tag} eval complete:[/]"
                    f" [cyan]val_bpb[/]={format_float(q_val_bpb)}"
                    f" [cyan]compressed_bytes[/]={qf.stat().st_size:,}"
                    f" in {q_eval_time:.1f}s"
                )

            quant_results[quant_tag] = {
                "scheme": quant_tag,
                "compressed_bytes": qf.stat().st_size,
                "val_bpb": q_val_bpb,
                "val_loss": q_val_loss,
                "val_tokens_evaluated": q_eval_tokens,
                "eval_time_seconds": round(q_eval_time, 2),
            }

        report["quant"] = quant_results

    if is_main_process():
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
