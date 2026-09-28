"""Render the measured BANKING77 adapter-free to task-LoRA comparison.

The report and its text-free row evidence are the source of truth. The SVG,
PNG, and provenance JSON are create-only presentation assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cairosvg


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/worthify/foundation-banking77-unadapted-transfer-20260927-v1/report.json"
STEM = ROOT / "docs/assets/decision-1/banking77-before-after-20260927-v1"
SCHEMA = "worthify-banking77-before-after-chart-v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source(report_path: Path) -> tuple[dict, bytes]:
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("test_rows") != 3080 or set(report.get("arms", {})) != {"raw", "tuned"}:
        raise ValueError("Expected the two-arm, 3,080-row held-out report")
    for arm, selected_correct in (("raw", 2881), ("tuned", 2871)):
        values = report["arms"][arm]
        before, after = values["zero_shot_correct"], values["lora_correct"]
        if not (isinstance(before, int) and 0 <= before < after <= 3080
                and after == selected_correct):
            raise ValueError(f"Unexpected {arm} correct counts")
        for field, expected in (("zero_shot_accuracy", before / 3080),
                                ("lora_accuracy", after / 3080),
                                ("lift", (after - before) / 3080)):
            if abs(values[field] - expected) > 1e-12:
                raise ValueError(f"{arm} {field} differs from counts")
    return report, report_bytes


def svg_bytes(report: dict, report_digest: str) -> bytes:
    x0, x1, lower, upper = 292, 960, 0.75, 1.0

    def x(value: float) -> float:
        if not lower <= value <= upper:
            raise ValueError("Accuracy falls outside the visibly labeled axis")
        return x0 + (x1 - x0) * (value - lower) / (upper - lower)

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="660" viewBox="0 0 1200 660" role="img" aria-labelledby="title desc">',
        '<title id="title">BANKING77 accuracy before and after a task LoRA</title>',
        '<desc id="desc">Both original Gemma and Decision-1 improve from their adapter-free scores after task LoRA training on the same 3,080 held-out 16-option BANKING77 rows. The axis begins at 75 percent and ends at 100 percent. Original Gemma retains the higher final LoRA score.</desc>',
        f'<metadata>schema={SCHEMA}; report_sha256={report_digest}</metadata>',
        '<rect width="1200" height="660" fill="#101b2a"/>',
        '<text x="72" y="58" fill="#f5ad82" font-family="Arial,Helvetica,sans-serif" font-size="19" font-weight="700" letter-spacing="3">WORTHIFY / DECISION-1</text>',
        '<text x="72" y="119" fill="#f7f8fa" font-family="Arial,Helvetica,sans-serif" font-size="43" font-weight="700">What the task LoRA added</text>',
        '<text x="72" y="155" fill="#c9d4e3" font-family="Arial,Helvetica,sans-serif" font-size="20">BANKING77 · 3,080 official-test utterances · 16 offered choices</text>',
        '<circle cx="292" cy="205" r="8" fill="#101b2a" stroke="#e1e8f0" stroke-width="3"/>',
        '<text x="309" y="212" fill="#c9d4e3" font-family="Arial,Helvetica,sans-serif" font-size="17">Without task LoRA</text>',
        '<circle cx="505" cy="205" r="8" fill="#f5ad82"/>',
        '<text x="522" y="212" fill="#c9d4e3" font-family="Arial,Helvetica,sans-serif" font-size="17">Validation-selected task LoRA</text>',
    ]
    for tick in (0.75, 0.80, 0.85, 0.90, 0.95, 1.0):
        position = x(tick)
        parts.extend((
            f'<line x1="{position:.1f}" y1="254" x2="{position:.1f}" y2="516" stroke="#3a4a60" stroke-width="1"/>',
            f'<text x="{position:.1f}" y="246" text-anchor="middle" fill="#aebdd0" font-family="Arial,Helvetica,sans-serif" font-size="15">{tick * 100:.0f}%</text>',
        ))
    parts.append('<line x1="72" y1="418" x2="1128" y2="418" stroke="#3a4a60" stroke-width="1"/>')
    for arm, name, y, color in (("tuned", "Decision-1", 335, "#f5ad82"),
                                 ("raw", "Original Gemma", 475, "#8fc5e8")):
        values = report["arms"][arm]
        before, after = values["zero_shot_accuracy"], values["lora_accuracy"]
        start, end = x(before), x(after)
        gain = (after - before) * 100
        parts.extend((
            f'<text x="72" y="{y - 5}" fill="#f7f8fa" font-family="Arial,Helvetica,sans-serif" font-size="25" font-weight="700">{name}</text>',
            f'<text x="72" y="{y + 24}" fill="#aebdd0" font-family="Arial,Helvetica,sans-serif" font-size="16">Same test rows</text>',
            f'<line x1="{start:.1f}" y1="{y}" x2="{end:.1f}" y2="{y}" stroke="{color}" stroke-width="8" stroke-linecap="round"/>',
            f'<circle cx="{start:.1f}" cy="{y}" r="11" fill="#101b2a" stroke="#e1e8f0" stroke-width="4"/>',
            f'<circle cx="{end:.1f}" cy="{y}" r="12" fill="{color}"/>',
            f'<text x="{start:.1f}" y="{y - 22}" text-anchor="middle" fill="#dce5ee" font-family="Arial,Helvetica,sans-serif" font-size="18" font-weight="700">{pct(before)}</text>',
            f'<text x="{end:.1f}" y="{y + 39}" text-anchor="middle" fill="{color}" font-family="Arial,Helvetica,sans-serif" font-size="19" font-weight="700">{pct(after)}</text>',
            f'<text x="1014" y="{y + 5}" fill="{color}" font-family="Arial,Helvetica,sans-serif" font-size="28" font-weight="700">+{gain:.2f} pp</text>',
        ))
    parts.extend((
        '<text x="72" y="559" fill="#c9d4e3" font-family="Arial,Helvetica,sans-serif" font-size="17">16-option test accuracy; horizontal axis begins at 75%.</text>',
        '<text x="72" y="595" fill="#aebdd0" font-family="Arial,Helvetica,sans-serif" font-size="16">Adapter-free scores were measured post-hoc. Within-base gains do not establish a stronger starting base.</text>',
        '<text x="72" y="624" fill="#aebdd0" font-family="Arial,Helvetica,sans-serif" font-size="16">In the 388-update study, selected Gemma and Decision-1 LoRAs scored 93.54% and 93.21%.</text>',
        '</svg>',
    ))
    return ("\n".join(parts) + "\n").encode()


def expected(report_path: Path) -> tuple[bytes, bytes, dict]:
    report, report_bytes = source(report_path)
    drawing = svg_bytes(report, sha(report_bytes))
    raster = cairosvg.svg2png(bytestring=drawing, output_width=1200, output_height=660)
    metadata = {
        "schema": SCHEMA,
        "report_sha256": sha(report_bytes),
        "plot_source_sha256": sha(Path(__file__).read_bytes()),
        "svg_sha256": sha(drawing),
        "png_sha256": sha(raster),
        "renderer": f"CairoSVG {cairosvg.__version__}",
        "test_rows": 3080,
        "axis_min_accuracy": 0.75,
        "axis_max_accuracy": 1.0,
    }
    return drawing, raster, metadata


def run(command: str, report_path: Path, stem: Path) -> dict:
    drawing, raster, metadata = expected(report_path)
    paths = {"svg": stem.with_suffix(".svg"), "png": stem.with_suffix(".png"),
             "json": stem.with_suffix(".json")}
    if command == "stage":
        if any(path.exists() for path in paths.values()):
            raise FileExistsError("Chart assets are create-only")
        stem.parent.mkdir(parents=True, exist_ok=True)
        for kind, content in (("svg", drawing), ("png", raster),
                              ("json", (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())):
            with paths[kind].open("xb") as stream:
                stream.write(content)
    elif command == "verify":
        if (paths["svg"].read_bytes() != drawing or paths["png"].read_bytes() != raster
                or json.loads(paths["json"].read_text()) != metadata):
            raise ValueError("Chart differs from report or deterministic rendering")
    else:
        raise ValueError(command)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "verify"))
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--stem", type=Path, default=STEM)
    args = parser.parse_args()
    print(json.dumps(run(args.command, args.report, args.stem), sort_keys=True))
