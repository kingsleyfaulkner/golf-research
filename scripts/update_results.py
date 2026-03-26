"""Update RESULTS.md from experiment artifact folders.

Scans all experiment directories for artifacts_* folders, extracts
summary statistics, and generates a markdown results table grouped
by GPU architecture and sorted by best quantized val_bpb ascending.

Each quantization scheme (int4-int8, mxfp4, nvfp4) gets its own BPB
column with a green/red indicator showing whether the compressed model
fits under the 16 MB contest limit.

Usage:
    python scripts/update_results.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Parameter Golf size limit: 16,000,000 bytes (decimal)
SIZE_LIMIT = 16_000_000


def parse_gpu_tag_from_system(system_path: Path) -> str | None:
    """Extract GPU tag from system.json (world_size + normalised GPU name)."""
    if not system_path.exists():
        return None
    with open(system_path) as f:
        info = json.load(f)
    gpus = info.get("gpus", [])
    world_size = info.get("distributed", {}).get("world_size", 1)
    if not gpus:
        return None
    name = gpus[0]["name"].upper().replace("NVIDIA", "").strip()
    name = re.sub(r"\d+\s*GB.*", "", name).strip()
    name = re.sub(r"BLACKWELL.*", "", name).strip()
    name = re.sub(r"SERVER.*", "", name).strip()
    name = re.sub(r"EDITION.*", "", name).strip()
    name = re.sub(r"GEFORCE\s*", "", name).strip()
    name = re.sub(r"\s+", "_", name).lower().strip("_")
    return f"{world_size}x_{name}"


def parse_gpu_tag_from_folder(folder_name: str) -> str:
    """Fallback: extract GPU tag from folder name, stripping run index suffix."""
    match = re.match(r"artifacts_(.+)", folder_name)
    if not match:
        return folder_name
    tag = match.group(1)
    tag = re.sub(r"_(\d{1,2})$", "", tag)
    return tag


def format_gpu_heading(tag: str) -> str:
    """Convert '2x_rtx_pro_6000' to '2x RTX PRO 6000'."""
    parts = tag.split("_", 1)
    count = parts[0]
    name = parts[1].replace("_", " ").upper() if len(parts) > 1 else ""
    return f"{count} {name}".strip()


def get_final_loss(metrics_path: Path) -> float | None:
    """Read the last loss value from metrics.jsonl."""
    last_line = None
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return None
    record = json.loads(last_line)
    return record.get("ce_loss") or record.get("loss")


def get_steps(metrics_path: Path) -> int | None:
    """Read the final step count from metrics.jsonl."""
    last_line = None
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return None
    record = json.loads(last_line)
    return record.get("global_step")


def get_tokens(metrics_path: Path) -> int | None:
    """Read the final tokens_seen count from metrics.jsonl."""
    last_line = None
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return None
    record = json.loads(last_line)
    return record.get("tokens_seen")


def get_run_datetime(artifacts_dir: Path) -> str | None:
    """Extract the run date/time from eval_report.json or file timestamps."""
    eval_path = artifacts_dir / "eval_report.json"
    if eval_path.exists():
        with open(eval_path) as f:
            report = json.load(f)
        ts = report.get("timestamp")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                return dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                pass

    system_path = artifacts_dir / "system.json"
    if system_path.exists():
        mtime = system_path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")

    mtime = artifacts_dir.stat().st_mtime
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M")


def collect_results(root: Path) -> list[dict]:
    """Scan all experiment folders for artifact results."""
    results = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        for artifacts_dir in sorted(experiment_dir.iterdir()):
            if not artifacts_dir.is_dir() or not artifacts_dir.name.startswith("artifacts_"):
                continue

            system_path = artifacts_dir / "system.json"
            gpu_tag = parse_gpu_tag_from_system(system_path) or parse_gpu_tag_from_folder(
                artifacts_dir.name
            )
            experiment = experiment_dir.name
            rel_path = f"experiments/{experiment}/{artifacts_dir.name}"

            entry = {
                "experiment": experiment,
                "gpu_tag": gpu_tag,
                "gpu_heading": format_gpu_heading(gpu_tag),
                "rel_path": rel_path,
                "steps": None,
                "tokens": None,
                "loss": None,
                "val_loss": None,
                "val_bpb": None,
                # keyed as "int4", "int6", etc: {"bpb": float, "compressed_bytes": int}
                "quant_results": {},
                "run_date": None,
            }

            metrics_path = artifacts_dir / "metrics.jsonl"
            if metrics_path.exists():
                entry["loss"] = get_final_loss(metrics_path)
                entry["steps"] = get_steps(metrics_path)
                entry["tokens"] = get_tokens(metrics_path)

            eval_path = artifacts_dir / "eval_report.json"
            if eval_path.exists():
                with open(eval_path) as f:
                    report = json.load(f)
                entry["val_loss"] = report.get("val_loss")
                entry["val_bpb"] = report.get("val_bpb")

                # Multi-level quant results (keyed as "int4", "int6", etc.)
                qr = report.get("quant", {}) or report.get("quant_results", {})
                for key, qdata in qr.items():
                    entry["quant_results"][key] = {
                        "bpb": qdata.get("val_bpb"),
                        "compressed_bytes": qdata.get("compressed_bytes")
                        or qdata.get("file_bytes"),
                    }

                # Backwards compat: old single-level format
                if not qr and report.get("quant_val_bpb") is not None:
                    bits = report.get("quant_bits", 8)
                    entry["quant_results"][f"int{bits}"] = {
                        "bpb": report["quant_val_bpb"],
                        "compressed_bytes": report.get("quant_file_bytes"),
                    }

            # Merge compressed_bytes from quant_report.json (may not have eval yet)
            quant_report_path = artifacts_dir / "quant_report.json"
            if quant_report_path.exists():
                with open(quant_report_path) as f:
                    qreport = json.load(f)
                for key, qdata in qreport.get("quant", {}).items():
                    if key not in entry["quant_results"]:
                        entry["quant_results"][key] = {"bpb": None, "compressed_bytes": None}
                    if entry["quant_results"][key].get("compressed_bytes") is None:
                        entry["quant_results"][key]["compressed_bytes"] = qdata.get(
                            "compressed_bytes"
                        )

            entry["run_date"] = get_run_datetime(artifacts_dir)
            results.append(entry)

    return results


def sort_key(r: dict) -> float:
    """Sort by val_bpb ascending, falling back to best quant BPB."""
    if r["val_bpb"] is not None:
        return r["val_bpb"]
    bpbs = [qd["bpb"] for qd in r["quant_results"].values() if qd.get("bpb") is not None]
    if bpbs:
        return min(bpbs)
    return float("inf")


def fmt(value, precision=4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def fmt_tokens(value) -> str:
    """Format token count in millions (e.g. 88.5M)."""
    if value is None:
        return "-"
    return f"{value / 1_000_000:.1f}M"


def fmt_bpb_with_size(bpb: float | None, compressed_bytes: int | None) -> str:
    """Format BPB with green/red size indicator based on 16MB limit."""
    if bpb is None and compressed_bytes is not None:
        size_mb = compressed_bytes / 1_000_000
        indicator = "\U0001f7e2" if compressed_bytes < SIZE_LIMIT else "\U0001f534"
        return f"{indicator} - ({size_mb:.1f}M)"
    if bpb is None:
        return "-"
    bpb_str = f"{bpb:.4f}"
    if compressed_bytes is not None:
        size_mb = compressed_bytes / 1_000_000
        indicator = "\U0001f7e2" if compressed_bytes < SIZE_LIMIT else "\U0001f534"
        return f"{indicator} {bpb_str} ({size_mb:.1f}M)"
    return bpb_str


def generate_markdown(results: list[dict]) -> str:
    """Generate RESULTS.md content from collected results."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["gpu_heading"], []).append(r)

    all_quant_keys: set[str] = set()
    for r in results:
        all_quant_keys.update(r["quant_results"].keys())
    quant_keys = sorted(all_quant_keys)

    lines = ["# Results", ""]

    for heading in sorted(groups.keys()):
        entries = groups[heading]
        entries.sort(key=sort_key)

        lines.append(f"## {heading}")
        lines.append("")

        # Build header
        header_cols = ["Experiment", "Date", "Steps", "Tokens", "Loss", "Val Loss", "Val BPB"]
        align_cols = [":---", ":---", "---:", "---:", "---:", "---:", "---:"]
        for qk in quant_keys:
            header_cols.append(f"{qk} BPB")
            align_cols.append("---:")
        header_cols.append("Artifacts")
        align_cols.append(":---")

        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join(align_cols) + " |")

        for r in entries:
            link = f"[artifacts]({r['rel_path']})"
            row_cols = [
                r["experiment"],
                r["run_date"] or "-",
                fmt(r["steps"]),
                fmt_tokens(r["tokens"]),
                fmt(r["loss"]),
                fmt(r["val_loss"]),
                fmt(r["val_bpb"]),
            ]
            for qk in quant_keys:
                qdata = r["quant_results"].get(qk, {})
                row_cols.append(fmt_bpb_with_size(qdata.get("bpb"), qdata.get("compressed_bytes")))
            row_cols.append(link)
            lines.append("| " + " | ".join(row_cols) + " |")

        lines.append("")

    return "\n".join(lines)


def regenerate_artifact_readmes(root: Path) -> None:
    """Call gen_artifact_readme for every artifact_* directory."""
    spec = importlib.util.spec_from_file_location(
        "gen_artifact_readme", root / "scripts" / "gen_artifact_readme.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    experiments_dir = root / "experiments"
    for experiment_dir in sorted(experiments_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        for artifact_dir in sorted(experiment_dir.iterdir()):
            if not artifact_dir.is_dir() or not artifact_dir.name.startswith("artifacts_"):
                continue
            old_argv = sys.argv
            try:
                sys.argv = [
                    "gen_artifact_readme.py",
                    str(artifact_dir),
                    str(experiment_dir),
                    str(root),
                ]
                mod.main()
            finally:
                sys.argv = old_argv
            print(f"  Regenerated {artifact_dir.relative_to(root)}")


def main():
    root = Path(__file__).resolve().parent.parent
    experiments_dir = root / "experiments"

    print("Regenerating artifact READMEs and charts...")
    regenerate_artifact_readmes(root)

    results = collect_results(experiments_dir)
    if not results:
        print("No artifact folders found")
        return

    md = generate_markdown(results)
    results_path = root / "RESULTS.md"
    results_path.write_text(md)
    print(f"Updated {results_path} with {len(results)} result(s)")


if __name__ == "__main__":
    main()
