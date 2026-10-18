"""Build the HW1 report with the unmodified official NeurIPS 2025 style.

Requires measured artifacts in results/, matplotlib, and Tectonic. The LaTeX
source is chest_xray_vae_report.tex. No experiment is rerun by this script.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


def build_loss_plot(history: list[dict[str, str]], output: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    train = [float(row["train_elbo"]) for row in history]
    val = [float(row["val_elbo"]) for row in history]
    image = Image.new("RGB", (1060, 330), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    if font_path.is_file():
        font = ImageFont.truetype(str(font_path), 19)
        small = ImageFont.truetype(str(font_path), 16)
    else:
        font = ImageFont.load_default(size=19)
        small = ImageFont.load_default(size=16)
    left, right, top, bottom = 118, 1032, 52, 268
    low = min(train + val) * 0.98
    high = max(train + val) * 1.01
    x_at = lambda i: left + (i - 1) * (right - left) / (len(epochs) - 1)
    y_at = lambda v: bottom - (v - low) * (bottom - top) / (high - low)
    for i in range(4):
        value = low + i * (high - low) / 3
        y = y_at(value)
        draw.line([(left, y), (right, y)], fill="#d8d8d8", width=1)
        draw.text((left - 8, y), f"{value:.0f}", fill="black", font=small,
                  anchor="rm")
    draw.line([(left, top), (left, bottom), (right, bottom)],
              fill="black", width=2)
    for epoch in (1, 5, 10, 15, 20):
        x = x_at(epoch)
        draw.text((x, bottom + 11), str(epoch), fill="black", font=small,
                  anchor="mt")
    draw.line([(x_at(i), y_at(v)) for i, v in zip(epochs, train)],
              fill="#163b66", width=4, joint="curve")
    draw.line([(x_at(i), y_at(v)) for i, v in zip(epochs, val)],
              fill="#b14f2f", width=4, joint="curve")
    draw.text(((left + right) / 2, 318), "Epoch", fill="black", font=font,
              anchor="mb")
    draw.text((12, 10), "Negative ELBO (nats/image)", fill="black", font=font)
    draw.line([(710, 22), (750, 22)], fill="#163b66", width=4)
    draw.text((760, 22), "Training", fill="black", font=small, anchor="lm")
    draw.line([(855, 22), (895, 22)], fill="#b14f2f", width=4)
    draw.text((905, 22), "Validation", fill="black", font=small, anchor="lm")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/pdf/chest_xray_vae_report.pdf")
    parser.add_argument("--tectonic", default=os.environ.get("TECTONIC"))
    args = parser.parse_args()
    results = args.results.resolve()
    output = args.output.resolve()
    training = json.loads((results / "training.json").read_text())
    evaluation = json.loads((results / "evaluation.json").read_text())
    with (results / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    if len(history) != 20:
        raise ValueError("The report describes the measured 20-epoch run")
    if abs(training["best_val_elbo"] - 2388.1038224377394) > 0.005:
        raise ValueError("Report values do not match training.json")
    if abs(evaluation["fid_1000_ignite"] - 0.1771436870130049) > 0.0005:
        raise ValueError("Report values do not match evaluation.json")
    if abs(evaluation["inception_score"] - 1.7791648601369754) > 0.0005:
        raise ValueError("Report values do not match evaluation.json")
    for name in ("generated_grid.png", "real_grid.png"):
        if not (results / name).is_file():
            raise FileNotFoundError(results / name)
    if results != ROOT / "results":
        raise ValueError("The LaTeX source refers to figures in the root results directory")

    build_loss_plot(history, results / "loss_curve.png")
    tectonic = args.tectonic or shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError("Install Tectonic or pass --tectonic /path/to/tectonic")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TECTONIC_CACHE_DIR", str(ROOT / "tmp/tectonic-cache"))
    subprocess.run([tectonic, "--outdir", str(output.parent),
                    str(ROOT / "chest_xray_vae_report.tex")],
                   cwd=ROOT, env=env, check=True)
    built = output.parent / "chest_xray_vae_report.pdf"
    if built != output:
        shutil.move(built, output)
    print(output)


if __name__ == "__main__":
    main()
