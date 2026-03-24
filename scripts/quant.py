"""Post-training quantization with compression.

Loads a Composer checkpoint, applies quantization, and saves compressed
artifacts. Supports multiple quantization schemes:

  - int4..int8: Per-row symmetric integer quantization with bit-packing
  - mxfp4: OCP Microscaling FP4 (E2M1 values, E8M0 block scales, block=32)
  - nvfp4: NVIDIA FP4 (E2M1 values, E4M3 block scales + FP32 tensor scale, block=16)

The quantized files are saved inside the checkpoint directory with the
scheme encoded in the name (e.g. ``step_2823_final_model_int6``,
``step_2823_final_model_mxfp4``).

Usage:
    python quant.py                             # default int8
    python quant.py --schemes int6,int8,mxfp4   # multiple schemes
    python quant.py --schemes mxfp4,nvfp4       # FP4 variants
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import zstandard

    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

import click
import numpy as np
import torch
from composer.nn.architecture.model import Model
from composer.utils.logging import setup_logging
from torch import Tensor

logger = logging.getLogger(__name__)

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
KEEP_FLOAT_MAX_NUMEL = 65_536
KEEP_FLOAT_STORE_DTYPE = torch.float16
PER_ROW_SCALE_DTYPE = torch.float16
CLIP_PERCENTILE = 99.99984
CLIP_Q = CLIP_PERCENTILE / 100.0

# All supported scheme names
SUPPORTED_SCHEMES = {"int4", "int5", "int6", "int7", "int8", "mxfp4", "nvfp4"}

# intN clamp ranges
QUANT_RANGES = {4: 7, 5: 15, 6: 31, 7: 63, 8: 127}

# FP4 E2M1 lookup table (indices 0-15, sign bit is MSB)
FP4_E2M1_LUT = torch.tensor(
    [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
FP4_E2M1_MAX = 6.0
E8M0_BIAS = 127


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def compress_bytes(raw: bytes) -> bytes:
    if _COMPRESSOR == "zstd":
        return zstandard.ZstdCompressor(level=22).compress(raw)
    return zlib.compress(raw, level=9)


def decompress_bytes(blob: bytes) -> bytes:
    try:
        import zstandard as _zstd

        return _zstd.ZstdDecompressor().decompress(blob)
    except Exception:
        return zlib.decompress(blob)


def pack_nbits(q: Tensor, bits: int) -> Tensor:
    """Pack signed int8 values into a bit-packed uint8 tensor."""
    if bits == 8:
        return q
    offset = 1 << (bits - 1)
    flat = q.reshape(-1).numpy().astype(np.int16) + offset
    n = len(flat)
    vals = flat.astype(np.uint64)
    bit_matrix = np.zeros((n, bits), dtype=np.uint8)
    for b in range(bits):
        bit_matrix[:, b] = (vals >> b) & 1
    bit_stream = bit_matrix.reshape(-1)
    pad = (-len(bit_stream)) % 8
    if pad:
        bit_stream = np.concatenate([bit_stream, np.zeros(pad, dtype=np.uint8)])
    bit_stream = bit_stream.reshape(-1, 8)
    packed = np.zeros(len(bit_stream), dtype=np.uint8)
    for b in range(8):
        packed |= bit_stream[:, b].astype(np.uint8) << b
    return torch.from_numpy(packed)


def unpack_nbits(packed: Tensor, bits: int, numel: int) -> Tensor:
    """Unpack a bit-packed uint8 tensor back to signed int8 values."""
    if bits == 8:
        return packed
    packed_np = packed.numpy()
    offset = 1 << (bits - 1)
    bit_stream = np.zeros(len(packed_np) * 8, dtype=np.uint8)
    for b in range(8):
        bit_stream[b::8] = (packed_np >> b) & 1
    total_bits = numel * bits
    bit_stream = bit_stream[:total_bits].reshape(numel, bits)
    vals = np.zeros(numel, dtype=np.int32)
    for b in range(bits):
        vals += bit_stream[:, b].astype(np.int32) << b
    result = (vals - offset).astype(np.int8)
    return torch.from_numpy(result)


def quantize_e2m1(x: Tensor) -> Tensor:
    """Quantize float values to FP4 E2M1 (4-bit indices 0-15).

    Uses the three-step threshold approach from fouroversix (MIT HAN Lab):
    - |x| < 2:  round to nearest {0, 0.5, 1, 1.5} (half-integer precision)
    - 2 <= |x| < 4: round to nearest {2, 3} (integer precision)
    - |x| >= 4: round to nearest {4, 6} (even-integer precision)

    Returns uint8 tensor with FP4 indices (0-7 positive, 8-15 negative).
    """
    x32 = x.float()
    ax = x32.abs()

    step1 = torch.round(2.0 * ax) / 2.0  # half-integer rounding
    step2 = torch.round(ax)  # integer rounding
    step3 = 2.0 * torch.round(ax / 2.0)  # even-integer rounding
    step3 = step3.clamp_max(FP4_E2M1_MAX)

    mask1 = ax < 2.0
    mask2 = ax < 4.0
    quantized = step1 * mask1 + step2 * (~mask1 & mask2) + step3 * (~mask1 & ~mask2)

    # Map float values to FP4 indices using the LUT
    # Positive values: 0->0, 0.5->1, 1->2, 1.5->3, 2->4, 3->5, 4->6, 6->7
    # Build reverse lookup: value -> index
    pos_lut = FP4_E2M1_LUT[:8]
    # Find nearest index by broadcasting comparison
    diffs = (quantized.unsqueeze(-1) - pos_lut.to(quantized.device)).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)

    # Set sign bit (bit 3) for negative values
    indices = indices | ((x32 < 0).to(torch.uint8) << 3)
    return indices


def dequantize_e2m1(indices: Tensor) -> Tensor:
    """Dequantize FP4 E2M1 indices back to float values."""
    lut = FP4_E2M1_LUT.to(indices.device)
    return lut[indices.to(torch.long)]


def pack_fp4(indices: Tensor) -> Tensor:
    """Pack FP4 indices (uint8, one per element) into nibble pairs (2 per byte).

    Even-position values go in lower nibble, odd-position in upper nibble.
    Input numel must be even.
    """
    flat = indices.reshape(-1)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    lo = flat[0::2]
    hi = flat[1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_fp4(packed: Tensor, numel: int) -> Tensor:
    """Unpack nibble-packed FP4 back to uint8 indices."""
    flat = packed.reshape(-1)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    # Interleave: [lo0, hi0, lo1, hi1, ...]
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = lo
    out[1::2] = hi
    return out[:numel]


def quantize_mxfp4_tensor(t: Tensor, block_size: int = 32) -> dict[str, Tensor]:
    """Quantize a tensor using MXFP4 (OCP Microscaling).

    - Block size: 32 elements share one E8M0 (power-of-2) scale
    - Values quantized to FP4 E2M1, packed as nibble pairs

    Returns dict with 'blocks' (packed uint8) and 'scales' (int32, biased).
    """
    t32 = t.float().reshape(-1)

    # Pad to block boundary
    pad = (-t32.numel()) % block_size
    if pad:
        t32 = torch.cat([t32, torch.zeros(pad, dtype=torch.float32)])

    blocks_2d = t32.reshape(-1, block_size)
    num_blocks = blocks_2d.shape[0]

    # Compute per-block E8M0 scale (power-of-2 from max absolute value)
    block_amax = blocks_2d.abs().amax(dim=1).clamp_min(1e-12)

    # E8M0 shared exponent: floor(log2(amax)) - FP4_MAX_POW2
    # This is the standard OCP Microscaling approach from torchao.
    # The scale is a lower bound — values up to 1.5x the scale's range
    # are still representable by FP4 (since FP4 max = 6 = 1.5 * 2^2).
    amax_int32 = block_amax.view(torch.int32)
    extracted_exp = ((amax_int32 >> 23) & 0xFF).to(torch.int32) - 127
    scale_exp = extracted_exp - 2  # FP4 E2M1 max pow2 = 2

    scale_biased = (scale_exp + E8M0_BIAS).clamp(0, 255).to(torch.int32)

    # Convert biased exponent back to float for division
    scale_float = ((scale_biased << 23).view(torch.float32)).clamp_min(1e-45)

    # Scale and quantize
    scaled = blocks_2d / scale_float.unsqueeze(1)
    fp4_indices = quantize_e2m1(scaled)

    # Pack nibble pairs
    packed = pack_fp4(fp4_indices.reshape(-1)).reshape(num_blocks, block_size // 2)

    return {
        "blocks": packed,
        "scales": scale_biased,
        "num_elements": t.numel(),
        "orig_shape": list(t.shape),
    }


def dequantize_mxfp4_tensor(data: dict[str, Tensor], dtype: torch.dtype) -> Tensor:
    """Dequantize MXFP4 packed data back to a float tensor."""
    packed = data["blocks"]
    scale_biased = data["scales"]
    num_elements = data["num_elements"]
    orig_shape = data["orig_shape"]
    block_size = 32

    num_blocks = packed.shape[0]
    # Unpack nibbles
    indices = unpack_fp4(packed.reshape(-1), num_blocks * block_size)
    values = dequantize_e2m1(indices).reshape(num_blocks, block_size)

    # Apply E8M0 scaling
    scale_exp = (scale_biased.to(torch.int32) - E8M0_BIAS).unsqueeze(1)
    values = torch.ldexp(values, scale_exp.expand_as(values))

    return values.reshape(-1)[:num_elements].reshape(orig_shape).to(dtype)


def quantize_nvfp4_tensor(t: Tensor, block_size: int = 16) -> dict[str, Tensor]:
    """Quantize a tensor using NVFP4 (NVIDIA FP4).

    - Block size: 16 elements share one E4M3 scale
    - Per-tensor FP32 scale normalizes so block scales fit E4M3
    - Two-level scaling for better precision than MXFP4

    Returns dict with 'blocks' (packed uint8), 'block_scales' (float8_e4m3fn),
    and 'tensor_scale' (float32).
    """
    t32 = t.float().reshape(-1)

    # Pad to block boundary
    pad = (-t32.numel()) % block_size
    if pad:
        t32 = torch.cat([t32, torch.zeros(pad, dtype=torch.float32)])

    blocks_2d = t32.reshape(-1, block_size)
    num_blocks = blocks_2d.shape[0]

    # Level 1: per-tensor scale
    # sg = global_amax / (FP4_MAX * E4M3_MAX)
    E4M3_MAX = 448.0  # max representable E4M3 value
    global_amax = t32.abs().max().clamp_min(1e-12)
    tensor_scale = (global_amax / (FP4_E2M1_MAX * E4M3_MAX)).to(torch.float32)

    # Level 2: per-block E4M3 scale
    # sb = block_amax / (tensor_scale * FP4_MAX)
    block_amax = blocks_2d.abs().amax(dim=1).clamp_min(1e-12)
    block_scales_fp32 = block_amax / (tensor_scale * FP4_E2M1_MAX)
    block_scales_e4m3 = block_scales_fp32.to(torch.float8_e4m3fn)

    # Quantize: x / (sg * sb), then round to E2M1
    combined_scale = tensor_scale * block_scales_e4m3.float()
    scaled = blocks_2d / combined_scale.unsqueeze(1).clamp_min(1e-45)
    fp4_indices = quantize_e2m1(scaled)

    # Pack nibble pairs
    packed = pack_fp4(fp4_indices.reshape(-1)).reshape(num_blocks, block_size // 2)

    return {
        "blocks": packed,
        "block_scales": block_scales_e4m3,
        "tensor_scale": tensor_scale,
        "num_elements": t.numel(),
        "orig_shape": list(t.shape),
    }


def dequantize_nvfp4_tensor(data: dict[str, Tensor], dtype: torch.dtype) -> Tensor:
    """Dequantize NVFP4 packed data back to a float tensor."""
    packed = data["blocks"]
    block_scales = data["block_scales"]
    tensor_scale = data["tensor_scale"]
    num_elements = data["num_elements"]
    orig_shape = data["orig_shape"]
    block_size = 16

    num_blocks = packed.shape[0]
    indices = unpack_fp4(packed.reshape(-1), num_blocks * block_size)
    values = dequantize_e2m1(indices).reshape(num_blocks, block_size)

    # Apply two-level scaling: x_hat = sg * sb * fp4_value
    combined_scale = tensor_scale.float() * block_scales.float()
    values = values * combined_scale.unsqueeze(1)

    return values.reshape(-1)[:num_elements].reshape(orig_shape).to(dtype)


def quantize_state_dict(
    state_dict: dict[str, Tensor], scheme: str = "int8"
) -> tuple[dict[str, object], dict[str, int]]:
    """Quantize a state dict using the given scheme.

    Schemes: int4-int8 (per-row symmetric integer), mxfp4, nvfp4.
    """
    is_intN = scheme.startswith("int")
    bits = int(scheme[3:]) if is_intN else None
    quant_range = QUANT_RANGES.get(bits) if bits else None

    quantized: dict[str, Any] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "quantized_payload_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["quantized_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["quantized_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        orig_dtype = str(t.dtype).removeprefix("torch.")

        if is_intN:
            q, s = _quantize_intN_tensor(t, quant_range)
            meta: dict[str, object] = {}
            if s.ndim > 0:
                meta["scheme"] = "per_row"
                meta["axis"] = 0
            if bits < 8:
                meta["orig_shape"] = list(q.shape)
                meta["orig_numel"] = int(q.numel())
                # q = pack_nbits(q, bits)
            if meta:
                qmeta[name] = meta
            quantized[name] = q
            scales[name] = s
            dtypes[name] = orig_dtype
            stats["quantized_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

        elif scheme == "mxfp4":
            data = quantize_mxfp4_tensor(t)
            quantized[name] = data
            dtypes[name] = orig_dtype
            payload = tensor_nbytes(data["blocks"]) + tensor_nbytes(data["scales"])
            stats["quantized_payload_bytes"] += payload

        elif scheme == "nvfp4":
            data = quantize_nvfp4_tensor(t)
            quantized[name] = data
            dtypes[name] = orig_dtype
            payload = (
                tensor_nbytes(data["blocks"])
                + tensor_nbytes(data["block_scales"])
                + tensor_nbytes(data["tensor_scale"])
            )
            stats["quantized_payload_bytes"] += payload

    obj: dict[str, object] = {
        "__quant_format__": scheme,
        "__quant_scheme__": scheme,
        "quantized": quantized,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if is_intN:
        obj["__quant_bits__"] = bits
        obj["scales"] = scales
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def _quantize_intN_tensor(t: Tensor, quant_range: int) -> tuple[Tensor, Tensor]:
    """Per-row symmetric integer quantization with percentile clipping."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(quant_range)).clamp_min(1.0 / float(quant_range))
        q = (
            torch.clamp(torch.round(clipped / scale[:, None]), -quant_range, quant_range)
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(dtype=PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(
        clip_abs / float(quant_range) if clip_abs > 0 else 1.0, dtype=torch.float32
    )
    q = (
        torch.clamp(
            torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -quant_range, quant_range
        )
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


def dequantize_state_dict(obj: dict[str, object]) -> dict[str, Tensor]:
    """Reconstruct a full-precision state dict from a quantized object.

    Dispatches to the appropriate dequantization based on the scheme.
    """
    scheme = obj.get("__quant_scheme__") or obj.get("__quant_format__", "int8")
    is_intN = scheme.startswith("int")
    # bits = obj.get("__quant_bits__", 8) if is_intN else None

    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, q_data in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])

        if is_intN:
            q = q_data
            s = obj["scales"][name]
            meta = qmeta.get(name, {})
            if "orig_numel" in meta:
                # q = unpack_nbits(q, bits, meta["orig_numel"]).reshape(meta["orig_shape"])
                q = q.reshape(meta["orig_shape"])
            if meta.get("scheme") == "per_row" or s.ndim > 0:
                s = s.to(dtype=torch.float32)
                out[name] = (
                    (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1))))
                    .to(dtype=dtype)
                    .contiguous()
                )
            else:
                out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()

        elif scheme == "mxfp4":
            out[name] = dequantize_mxfp4_tensor(q_data, dtype)

        elif scheme == "nvfp4":
            out[name] = dequantize_nvfp4_tensor(q_data, dtype)

    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def quantize_one(
    model: torch.nn.Module,
    state_dict: dict[str, Tensor],
    composer_config: dict | None,
    scheme: str,
    output_path: Path,
) -> dict[str, Any]:
    """Quantize at a single scheme. Returns report dict."""
    logger.info(f"[bold]Quantizing to {scheme}[/]")
    quant_obj, quant_stats = quantize_state_dict(state_dict, scheme=scheme)

    if composer_config is not None:
        quant_obj["__composer_config__"] = composer_config

    # Serialize + compress
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = compress_bytes(quant_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(quant_blob)

    file_bytes = output_path.stat().st_size

    logger.info(
        f"[bold green]{scheme} ({_COMPRESSOR}):[/]"
        f" [cyan]compressed[/]={file_bytes:,} bytes"
        f" [cyan]serialized[/]={len(quant_raw):,} bytes"
        f" [cyan]payload[/]={quant_stats['quantized_payload_bytes']:,} bytes"
    )

    # Verify roundtrip
    with open(output_path, "rb") as f:
        verify_blob = f.read()
    verify_state = torch.load(
        io.BytesIO(decompress_bytes(verify_blob)),
        map_location="cpu",
        weights_only=False,
    )
    dequantized = dequantize_state_dict(verify_state)
    model.load_state_dict(dequantized, strict=True)
    logger.info(f"[bold green]{scheme} roundtrip verified[/]")

    return {
        "scheme": scheme,
        "compressor": _COMPRESSOR,
        "compressed_bytes": file_bytes,
        "serialized_bytes": len(quant_raw),
        "param_count": quant_stats["param_count"],
    }


@click.command()
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(),
    help="Path to checkpoint directory or .tar.gz (default: artifacts/checkpoint).",
)
@click.option(
    "--step",
    default=None,
    type=int,
    help="Specific checkpoint step to load (latest if omitted).",
)
@click.option(
    "--schemes",
    default="int8",
    type=str,
    help="Quantization scheme(s), comma-separated (default: int8). "
    "Supported: int4,int5,int6,int7,int8,mxfp4,nvfp4.",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(),
    help="Directory to write quantized files (default: same as checkpoint).",
)
def main(
    checkpoint: str | None,
    step: int | None,
    schemes: str,
    output_dir: str | None,
):
    """Quantize a Composer checkpoint."""
    setup_logging()

    # Parse schemes
    scheme_list = [s.strip() for s in schemes.split(",")]
    for s in scheme_list:
        if s not in SUPPORTED_SCHEMES:
            raise click.BadParameter(
                f"Unsupported scheme: {s}. Must be one of {sorted(SUPPORTED_SCHEMES)}"
            )

    # Resolve checkpoint
    tmp_dir = None
    tar_source_dir = None
    if checkpoint is None:
        checkpoint_path = Path("artifacts/checkpoint")
        if not checkpoint_path.exists():
            for candidate in [Path("artifacts/checkpoint.tar.gz"), Path("checkpoint.tar.gz")]:
                if candidate.exists():
                    checkpoint_path = candidate
                    break
            else:
                raise click.BadParameter(
                    "No checkpoint found. Looked for artifacts/checkpoint/"
                    " and artifacts/checkpoint.tar.gz"
                )
    else:
        checkpoint_path = Path(checkpoint)

    if not checkpoint_path.exists():
        raise click.BadParameter(f"Checkpoint not found: {checkpoint_path}")

    # If directory has only quant files, fall back to tar.gz
    if checkpoint_path.is_dir():
        full_candidates = [
            p
            for p in checkpoint_path.glob("step_*")
            if not re.search(r"_(int\d+|mxfp\d+|nvfp\d+)$", p.name)
        ]
        if not full_candidates:
            tar_fallback = checkpoint_path.parent / "checkpoint.tar.gz"
            if tar_fallback.exists():
                checkpoint_path = tar_fallback
            else:
                raise click.BadParameter(
                    f"No full-precision checkpoint in {checkpoint_path}"
                    " and no checkpoint.tar.gz found"
                )

    # Extract tar.gz if needed
    if checkpoint_path.suffix == ".gz" and checkpoint_path.name.endswith(".tar.gz"):
        logger.info(f"[bold]Extracting compressed checkpoint:[/] {checkpoint_path}")
        tar_source_dir = checkpoint_path.parent
        tmp_dir = tempfile.mkdtemp(prefix="quant_checkpoint_")
        with tarfile.open(checkpoint_path, "r:gz") as tar:
            tar.extractall(tmp_dir)
        extracted = Path(tmp_dir) / "checkpoint"
        if not extracted.exists():
            extracted = Path(tmp_dir)
        checkpoint_path = extracted

    # Resolve the actual checkpoint file
    resolved_checkpoint = checkpoint_path
    if checkpoint_path.is_dir():
        candidates = [
            p
            for p in sorted(checkpoint_path.glob("step_*"))
            if not re.search(r"_(int\d+|mxfp\d+|nvfp\d+)$", p.name)
        ]
        if not candidates:
            raise click.BadParameter(f"No full-precision checkpoint found in: {checkpoint_path}")
        resolved_checkpoint = max(candidates, key=lambda p: int(p.name.split("_")[1]))

    # Output directory
    if output_dir is not None:
        out_dir = Path(output_dir)
    elif tar_source_dir is not None:
        out_dir = tar_source_dir / "checkpoint"
    else:
        out_dir = resolved_checkpoint.parent

    # Load model once
    logger.info(f"[bold]Loading checkpoint:[/] {resolved_checkpoint}")
    model = Model.from_checkpoint(checkpoint_path, device="cpu", max_step=step)
    state_dict = model.state_dict()
    composer_config = state_dict.pop("__composer_config__", None)

    # Quantize each scheme
    quant = {}
    for scheme in scheme_list:
        out_path = out_dir / f"{resolved_checkpoint.name}_{scheme}"
        result = quantize_one(model, state_dict, composer_config, scheme, out_path)
        quant[scheme] = result

    # Merge into existing quant report (preserve results from previous runs)
    report_dir = out_dir.parent if out_dir.name == "checkpoint" else out_dir
    report_path = report_dir / "quant_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        report.setdefault("quant", {}).update(quant)
    else:
        report = {"quant": quant}
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    # Clean up
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
