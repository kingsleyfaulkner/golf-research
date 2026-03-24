"""Generate a README.md for an experiment artifact folder.

Combines the experiment description, runtime overrides, config diffs vs
baseline, results summary from metrics/eval/quant/system files, and
comparison against the latest baseline run on the same hardware.

Usage:
    python gen_artifact_readme.py <artifact_dir> <experiment_dir> <root_dir>
"""

import json
import math
import subprocess
import sys
from datetime import datetime
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


def read_all_metrics(path):
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_loss_chart_svg(exp_metrics, baseline_metrics=None):
    """Return SVG string for loss vs runtime (minutes) on a log-y scale."""
    W, H = 720, 400
    ML, MR, MT, MB = 65, 20, 20, 50

    pw = W - ML - MR
    ph = H - MT - MB

    def to_points(metrics):
        if not metrics:
            return []
        t0 = datetime.fromisoformat(metrics[0]["timestamp"])
        pts = []
        for m in metrics:
            try:
                t = datetime.fromisoformat(m["timestamp"])
                minutes = (t - t0).total_seconds() / 60.0
                loss = m.get("ce_loss") or m.get("loss")
                if loss and loss > 0:
                    pts.append((minutes, float(loss)))
            except Exception:
                pass
        return pts

    exp_pts = to_points(exp_metrics)
    base_pts = to_points(baseline_metrics) if baseline_metrics else []

    all_pts = exp_pts + base_pts
    if not all_pts:
        return None

    all_x = [p[0] for p in all_pts]
    all_y = [p[1] for p in all_pts]

    x_min, x_max = 0, max(all_x)
    y_min_data = min(all_y)
    y_max_data = max(all_y)

    # Pad log-scale bounds slightly
    log_lo = math.floor(math.log10(y_min_data) * 4) / 4 - 0.1
    log_hi = math.ceil(math.log10(y_max_data) * 4) / 4 + 0.1
    y_min = 10**log_lo
    y_max = 10**log_hi

    def sx(x):
        if x_max == x_min:
            return ML + pw / 2
        return ML + (x - x_min) / (x_max - x_min) * pw

    def sy(y):
        frac = (math.log10(y) - log_lo) / (log_hi - log_lo)
        return MT + (1 - frac) * ph

    def polyline(pts):
        return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)

    # Y ticks at 1, 2, 3, 5 per decade
    y_ticks = []
    for decade in range(math.floor(math.log10(y_min_data)) - 1, math.ceil(math.log10(y_max_data)) + 2):
        for mult in [1, 2, 3, 5]:
            val = mult * (10**decade)
            if y_min <= val <= y_max:
                y_ticks.append((val, mult == 1))

    # X ticks — roughly 5-6 ticks
    if x_max > 0:
        raw_step = x_max / 5
        mag = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
        step = max(1, round(raw_step / mag)) * mag
    else:
        step = 1
    x_ticks = []
    t = 0.0
    while t <= x_max + step * 0.01:
        x_ticks.append(t)
        t += step

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" style="background:white">'
    )
    svg.append(f'<rect x="{ML}" y="{MT}" width="{pw}" height="{ph}" fill="#f9f9f9"/>')

    # Grid lines
    for val, major in y_ticks:
        yy = sy(val)
        color = "#ccc" if major else "#e8e8e8"
        svg.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{ML+pw}" y2="{yy:.1f}" stroke="{color}" stroke-width="1"/>')
    for tx in x_ticks:
        xx = sx(tx)
        svg.append(f'<line x1="{xx:.1f}" y1="{MT}" x2="{xx:.1f}" y2="{MT+ph}" stroke="#e8e8e8" stroke-width="1"/>')

    # Data series
    if base_pts:
        svg.append(
            f'<polyline points="{polyline(base_pts)}" fill="none" '
            f'stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="5,3"/>'
        )
    if exp_pts:
        svg.append(
            f'<polyline points="{polyline(exp_pts)}" fill="none" '
            f'stroke="#2563eb" stroke-width="2"/>'
        )

    # Axes border
    svg.append(f'<rect x="{ML}" y="{MT}" width="{pw}" height="{ph}" fill="none" stroke="#555" stroke-width="1.5"/>')

    # Y tick marks and labels
    for val, major in y_ticks:
        yy = sy(val)
        if major:
            label = f"{val:.0f}" if val >= 10 else (f"{val:.1f}" if val >= 1 else f"{val:.2f}")
            svg.append(f'<line x1="{ML-5}" y1="{yy:.1f}" x2="{ML}" y2="{yy:.1f}" stroke="#555" stroke-width="1.5"/>')
            svg.append(
                f'<text x="{ML-8}" y="{yy:.1f}" text-anchor="end" dominant-baseline="middle" '
                f'font-size="11" font-family="monospace,sans-serif">{label}</text>'
            )
        else:
            svg.append(f'<line x1="{ML-3}" y1="{yy:.1f}" x2="{ML}" y2="{yy:.1f}" stroke="#888" stroke-width="1"/>')

    # X tick marks and labels
    for tx in x_ticks:
        xx = sx(tx)
        svg.append(f'<line x1="{xx:.1f}" y1="{MT+ph}" x2="{xx:.1f}" y2="{MT+ph+5}" stroke="#555" stroke-width="1.5"/>')
        svg.append(
            f'<text x="{xx:.1f}" y="{MT+ph+18}" text-anchor="middle" '
            f'font-size="11" font-family="monospace,sans-serif">{tx:.0f}</text>'
        )

    # Axis labels
    svg.append(
        f'<text x="{ML + pw // 2}" y="{H - 8}" text-anchor="middle" '
        f'font-size="13" font-family="sans-serif">Runtime (minutes)</text>'
    )
    cy = MT + ph // 2
    svg.append(
        f'<text transform="rotate(-90,16,{cy})" x="16" y="{cy}" text-anchor="middle" '
        f'font-size="13" font-family="sans-serif">Loss (log scale)</text>'
    )

    # Legend (only when baseline is shown)
    if base_pts:
        lx = ML + pw - 115
        ly = MT + 12
        svg.append(
            f'<rect x="{lx-5}" y="{ly-6}" width="120" height="46" '
            f'fill="white" fill-opacity="0.85" stroke="#ddd" stroke-width="1" rx="3"/>'
        )
        svg.append(
            f'<line x1="{lx}" y1="{ly+5}" x2="{lx+22}" y2="{ly+5}" '
            f'stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="5,3"/>'
        )
        svg.append(
            f'<text x="{lx+27}" y="{ly+9}" font-size="11" font-family="sans-serif">Baseline</text>'
        )
        svg.append(
            f'<line x1="{lx}" y1="{ly+24}" x2="{lx+22}" y2="{ly+24}" '
            f'stroke="#2563eb" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{lx+27}" y="{ly+28}" font-size="11" font-family="sans-serif">Experiment</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


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
    all_metrics = read_all_metrics(artifact_dir / "metrics.jsonl")
    eval_report = load_json(artifact_dir / "eval_report.json")
    quant_data = eval_report.get("quant", {}) if eval_report else {}

    # Generate and save loss chart SVG
    gpu_tag_for_chart = artifact_name.replace("artifacts_", "").rstrip("_0123456789")
    chart_baseline_metrics = None
    if experiment_name != "baseline":
        chart_baseline_artifact, _ = find_baseline(root_dir, gpu_tag_for_chart)
        if chart_baseline_artifact:
            chart_baseline_metrics = read_all_metrics(chart_baseline_artifact / "metrics.jsonl")
    svg_content = generate_loss_chart_svg(all_metrics, chart_baseline_metrics)
    if svg_content:
        (artifact_dir / "loss_chart.svg").write_text(svg_content)

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

    # Loss chart
    if svg_content:
        lines.append("")
        lines.append("## Loss Curve")
        lines.append("")
        lines.append("![Loss vs runtime (log scale)](loss_chart.svg)")

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
