"""Generate a README.md for an experiment artifact folder.

Combines the experiment description, runtime overrides, config diffs vs
baseline, results summary from metrics/eval/quant/system files, and
comparison against the latest baseline run on the same hardware.

Usage:
    python gen_artifact_readme.py <artifact_dir> <experiment_dir> <root_dir>
"""

import json
import subprocess
import sys
from pathlib import Path


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def last_metrics_line(path):
    if not path.exists():
        return None
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    return json.loads(last) if last else None


def fmt(val, precision=4):
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def fmt_size(nbytes):
    if nbytes is None:
        return "-"
    return f"{nbytes / 1_000_000:.1f} MB"


def find_baseline(root, gpu_tag):
    """Find the latest baseline artifacts dir and eval_report for the same GPU tag."""
    baseline_dir = root / "experiments" / "baseline"
    if not baseline_dir.exists():
        return None, None
    candidates = []
    for d in sorted(baseline_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("artifacts_"):
            continue
        tag = d.name.replace("artifacts_", "").rstrip("_0123456789")
        if tag == gpu_tag:
            candidates.append(d)
    if not candidates:
        return None, None
    baseline_artifact = candidates[-1]
    eval_report = load_json(baseline_artifact / "eval_report.json")
    return baseline_artifact, eval_report


def main():
    artifact_dir = Path(sys.argv[1])
    experiment_dir = Path(sys.argv[2])
    root_dir = Path(sys.argv[3])

    experiment_name = experiment_dir.name
    artifact_name = artifact_dir.name

    lines = []

    # Experiment description
    exp_readme = experiment_dir / "README.md"
    if exp_readme.exists():
        lines.append(exp_readme.read_text().rstrip())
    else:
        lines.append(f"# {experiment_name}")

    # Runtime overrides
    overrides_path = artifact_dir / "overrides.yaml"
    if overrides_path.exists():
        overrides = [
            line
            for line in overrides_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if overrides:
            lines.append("")
            lines.append("## Runtime Overrides")
            lines.append("")
            lines.append("```yaml")
            lines.extend(overrides)
            lines.append("```")

    # Load data sources
    system = load_json(artifact_dir / "system.json")
    metrics = last_metrics_line(artifact_dir / "metrics.jsonl")
    eval_report = load_json(artifact_dir / "eval_report.json")
    quant_data = eval_report.get("quant", {}) if eval_report else {}

    # Results summary (BPB first — most important)
    lines.append("")
    lines.append("## Results")
    lines.append("")

    if metrics:
        steps = metrics.get("global_step", "-")
        loss = fmt(metrics.get("ce_loss") or metrics.get("loss"))
        lines.append(f"- **Steps:** {steps}")
        lines.append(f"- **Train loss:** {loss}")

    if eval_report:
        lines.append(f"- **Val loss:** {fmt(eval_report.get('val_loss'))}")
        lines.append(f"- **Val BPB:** {fmt(eval_report.get('val_bpb'))}")

    # Comparison vs baseline (before quant details)
    if experiment_name != "baseline" and eval_report:
        gpu_tag = artifact_name.replace("artifacts_", "").rstrip("_0123456789")
        baseline_artifact, baseline_eval = find_baseline(root_dir, gpu_tag)
        if baseline_eval and baseline_artifact:
            rel_baseline = f"../../baseline/{baseline_artifact.name}"
            lines.append("")
            lines.append(f"## vs Baseline ([{baseline_artifact.name}]({rel_baseline}))")
            lines.append("")

            b_bpb = baseline_eval.get("val_bpb")
            e_bpb = eval_report.get("val_bpb")
            if b_bpb is not None and e_bpb is not None:
                delta = e_bpb - b_bpb
                sign = "+" if delta >= 0 else ""
                lines.append(f"- **Val BPB:** {fmt(e_bpb)} vs {fmt(b_bpb)} ({sign}{delta:.4f})")

            b_quant = baseline_eval.get("quant", {})
            e_quant = eval_report.get("quant", {})
            shared = sorted(set(b_quant) & set(e_quant))

            cols = ["full"] + shared
            lines.append("")
            lines.append("| | " + " | ".join(cols) + " |")
            lines.append("| :--- | " + " | ".join(["---:" for _ in cols]) + " |")
            exp_row = f"| **Experiment** | {fmt(e_bpb)} |"
            base_row = f"| **Baseline** | {fmt(b_bpb)} |"
            if e_bpb is not None and b_bpb is not None:
                d = e_bpb - b_bpb
                sign = "+" if d >= 0 else ""
                delta_row = f"| **Delta** | {sign}{d:.4f} |"
            else:
                delta_row = "| **Delta** | - |"
            for s in shared:
                eb = e_quant[s].get("val_bpb")
                bb = b_quant[s].get("val_bpb")
                exp_row += f" {fmt(eb)} |"
                base_row += f" {fmt(bb)} |"
                if eb is not None and bb is not None:
                    d = eb - bb
                    sign = "+" if d >= 0 else ""
                    delta_row += f" {sign}{d:.4f} |"
                else:
                    delta_row += " - |"
            lines.append(exp_row)
            lines.append(base_row)
            lines.append(delta_row)

    # Quant results as horizontal table (schemes across top)
    if quant_data:
        schemes = sorted(quant_data.keys())
        lines.append("")
        lines.append("## Quantization")
        lines.append("")
        lines.append("| | " + " | ".join(schemes) + " |")
        lines.append("| :--- | " + " | ".join(["---:" for _ in schemes]) + " |")
        bpb_row = "| **BPB** |"
        size_row = "| **Size** |"
        for s in schemes:
            qd = quant_data[s]
            bpb_row += f" {fmt(qd.get('val_bpb'))} |"
            size_row += f" {fmt_size(qd.get('compressed_bytes'))} |"
        lines.append(bpb_row)
        lines.append(size_row)

    # Config diff vs baseline
    if experiment_name != "baseline":
        baseline_dir = root_dir / "experiments" / "baseline"
        if baseline_dir.exists():
            diffs = []
            for fname in ["train.yaml", "model.yaml"]:
                base_f = baseline_dir / fname
                exp_f = experiment_dir / fname
                if base_f.exists() and exp_f.exists():
                    result = subprocess.run(
                        [
                            "bash",
                            "-c",
                            f"diff -u <(grep -v '^#' '{base_f}' | grep -v '^$') "
                            f"<(grep -v '^#' '{exp_f}' | grep -v '^$')",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    diff_text = "\n".join(result.stdout.splitlines()[2:])
                    if diff_text.strip():
                        diffs.append((fname, diff_text))

            if diffs:
                lines.append("")
                lines.append("## Config Changes vs Baseline")
                for fname, diff_text in diffs:
                    lines.append("")
                    lines.append(f"**{fname}:**")
                    lines.append("")
                    lines.append("```diff")
                    lines.append(diff_text)
                    lines.append("```")

    if system:
        gpus = system.get("gpus", [])
        dist = system.get("distributed", {})
        host = system.get("host", {})
        packages = system.get("packages", {})
        lines.append("")
        lines.append("## Platform")
        lines.append("")
        if gpus:
            gpu = gpus[0]
            lines.append(
                f"- **GPU:** {gpu.get('name', 'unknown')} ({gpu.get('total_memory_gb', '?')} GB)"
            )
            lines.append(f"- **GPUs:** {dist.get('world_size', len(gpus))}")
        cpu_name = host.get("cpu") or host.get("processor") or host.get("architecture", "")
        if cpu_name and cpu_name != "N/A":
            cpu_count = host.get("cpu_count")
            cpu_str = cpu_name
            if cpu_count:
                cpu_str += f" ({cpu_count} cores)"
            lines.append(f"- **CPU:** {cpu_str}")
        mem_gb = host.get("memory_gb")
        if mem_gb:
            lines.append(f"- **RAM:** {mem_gb:.0f} GB")
        torch_ver = packages.get("torch", "")
        cuda_ver = packages.get("cuda", "")
        if torch_ver:
            ver_str = f"PyTorch {torch_ver}"
            if cuda_ver:
                ver_str += f", CUDA {cuda_ver}"
            lines.append(f"- **Software:** {ver_str}")

    (artifact_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
