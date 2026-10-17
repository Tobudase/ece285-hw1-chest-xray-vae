"""Render an exactly two-page, NeurIPS-proportioned experiment report.

Reads measured results only; refuses to create a report without training and
evaluation artifacts. Run with the bundled/reportlab-enabled Python runtime.
"""

import argparse
import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


PAGE_W, PAGE_H = 612, 792
# NeurIPS 2025 specifies a 5.5 x 9 inch text block with 1.5 inch side margins.
MARGIN, GAP = 108, 12
TEXT_W = PAGE_W - 2 * MARGIN
COL_W = (TEXT_W - GAP) / 2
BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.5,
                      leading=11.4, alignment=TA_JUSTIFY, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.3, leading=10.1, alignment=TA_LEFT)
ABSTRACT = ParagraphStyle("abstract", parent=BODY, fontSize=9.4, leading=11.6)


def paragraph(c, html, x, y, width=TEXT_W, style=BODY):
    p = Paragraph(html, style)
    _, h = p.wrap(width, 1000)
    p.drawOn(c, x, y - h)
    return y - h - style.spaceAfter


def heading(c, title, x, y):
    y -= 6
    c.setFont("Times-Bold", 10.3)
    c.drawString(x, y, title)
    return y - 14


def line(c, x1, y1, x2, y2, color=colors.black, width=0.6):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)


def draw_table(c, rows, x, y, widths, font=8.4, row_height=14):
    total = sum(widths)
    line(c, x, y, x + total, y, width=0.8)
    for i, row in enumerate(rows):
        yy = y - (i + 1) * row_height + 4
        c.setFont("Times-Bold" if i == 0 else "Times-Roman", font)
        pos = x + 3
        for value, w in zip(row, widths):
            c.drawString(pos, yy, str(value))
            pos += w
        if i == 0:
            line(c, x, y - row_height, x + total, y - row_height, width=0.45)
    bottom = y - len(rows) * row_height
    line(c, x, bottom, x + total, bottom, width=0.8)
    return bottom - 8


def loss_plot(c, history, x, y, w=TEXT_W, h=130):
    left, right, bottom, top = x + 32, x + w - 7, y - h + 22, y - 9
    values = [float(r[k]) for r in history for k in ("train_elbo", "val_elbo")]
    low = min(values) * 0.98
    high = max(values) * 1.01
    if high <= low:
        high = low + 1
    line(c, left, bottom, right, bottom, width=0.5)
    line(c, left, bottom, left, top, width=0.5)
    for i in range(3):
        val = low + i * (high - low) / 2
        yy = bottom + i * (top - bottom) / 2
        c.setFont("Times-Roman", 7.4)
        c.drawRightString(left - 4, yy - 2, f"{val:.0f}")
        line(c, left, yy, right, yy, colors.lightgrey, 0.35)
    n = len(history)
    for key, color in (("train_elbo", colors.HexColor("#163b66")),
                       ("val_elbo", colors.HexColor("#b14f2f"))):
        c.setStrokeColor(color)
        c.setLineWidth(1.5)
        for i, row in enumerate(history):
            xx = left + i * (right - left) / max(n - 1, 1)
            yy = bottom + (float(row[key]) - low) * (top - bottom) / (high - low)
            if i:
                c.line(last_x, last_y, xx, yy)
            last_x, last_y = xx, yy
    c.setFont("Times-Roman", 7.5)
    c.setFillColor(colors.black)
    c.drawString(left, bottom - 12, "1")
    c.drawRightString(right, bottom - 12, str(n))
    c.drawCentredString((left + right) / 2, bottom - 12, "epoch")
    c.setFont("Times-Roman", 7.8)
    c.setFillColor(colors.HexColor("#163b66")); c.drawString(left + 3, top + 2, "train")
    c.setFillColor(colors.HexColor("#b14f2f")); c.drawString(left + 45, top + 2, "validation")
    c.setFillColor(colors.black)
    return y - h - 5


def figure(c, path, x, y, label, side=COL_W):
    img = ImageReader(str(path))
    iw, ih = img.getSize()
    c.drawImage(img, x, y - side, width=side, height=side * ih / iw,
                preserveAspectRatio=True, anchor="c")
    yy = y - side * ih / iw - 11
    c.setFont("Times-Bold", 8.5)
    c.drawString(x, yy, label)
    return yy - 13


def footer(c, page):
    line(c, MARGIN, 42, PAGE_W - MARGIN, 42, colors.HexColor("#bbbbbb"), 0.4)
    c.setFont("Times-Roman", 8)
    c.drawCentredString(PAGE_W / 2, 29, str(page))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--output", default="output/pdf/chest_xray_vae_report.pdf")
    args = p.parse_args()
    root = Path(args.results)
    train = json.loads((root / "training.json").read_text())
    evaluation = json.loads((root / "evaluation.json").read_text())
    history = list(csv.DictReader((root / "history.csv").open()))
    if not history or not (root / "generated_grid.png").exists() or not (root / "real_grid.png").exists():
        raise RuntimeError("Measured training, evaluation, and example artifacts are required")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("Generating Chest Radiographs with a Convolutional VAE")
    c.setAuthor("ECE 285 Homework 1")

    # Page 1: NeurIPS-sized single-column text block.
    c.setFont("Times-Bold", 17)
    c.drawCentredString(PAGE_W / 2, 718, "Generating Chest Radiographs with a")
    c.drawCentredString(PAGE_W / 2, 697, "Convolutional Variational Autoencoder")
    c.setFont("Times-Roman", 10)
    c.drawCentredString(PAGE_W / 2, 677, "ECE 285 - Homework 1")
    line(c, MARGIN, 663, PAGE_W - MARGIN, 663, colors.HexColor("#bbbbbb"), 0.5)
    c.setFont("Times-Bold", 9.5)
    c.drawString(MARGIN, 650, "Abstract")
    abstract = ("A convolutional variational autoencoder (VAE) was trained on pediatric "
                "chest radiographs from the Kaggle pneumonia dataset. We report the "
                "reconstruction and KL losses, unconditional prior samples, and Ignite "
                "Inception Score (IS) and Fréchet Inception Distance (FID) against held-out "
                "test radiographs. The measured scores describe image statistics only; "
                "they do not establish clinical fidelity.")
    y = paragraph(c, abstract, MARGIN, 640, TEXT_W, ABSTRACT)
    x = MARGIN
    y = heading(c, "1  Data and preprocessing", x, y - 3)
    y = paragraph(c, "We used the <i>Chest X-Ray Images (Pneumonia)</i> dataset [1]. "
                   f"Its original training split was divided by class into {train['train_count']} "
                   f"training and {train['val_count']} validation images with seed {train['seed']}. "
                   f"The separate test split ({evaluation['test_total']} images) was reserved "
                   "for generative evaluation. We converted each image to one grayscale "
                   "channel, resized it to 64 x 64 pixels, and scaled values to [0, 1].", x, y)
    y = heading(c, "2  VAE and objective", x, y)
    y = paragraph(c, "Four stride-2 convolutional layers map an image to 4 x 4 x 128 "
                   f"features. Two linear heads produce the mean and log variance of a "
                   f"{train['latent_dim']}-dimensional Gaussian posterior. A reparameterized "
                   "sample enters a linear layer and four transposed convolutions. A "
                   f"sigmoid decoder outputs grayscale pixels. The model has "
                   f"{train['parameter_count']:,} trainable parameters [2].", x, y)
    y = paragraph(c, "We minimized pixelwise binary cross entropy summed over all "
                   "pixels plus beta times KL(q(z|x) || N(0,I)), averaged per image. "
                   "The KL term is analytic; samples use the reparameterization trick. "
                   "For generation, z is drawn from the unit Gaussian prior, without "
                   "using a test image.", x, y)
    y = heading(c, "3  Training configuration", x, y)
    y = draw_table(c, [
        ("Setting", "Value"),
        ("Image / latent", f"64 x 64 / {train['latent_dim']}"),
        ("Batch / epochs", f"{train['batch_size']} / {len(history)}"),
        ("Optimizer / learning rate", f"Adam / {train['lr']:g}"),
        ("KL weight beta", f"{train['beta']:g}"),
        ("Device / seed", f"{train['device_used']} / {train['seed']}"),
    ], x, y, [190, TEXT_W - 190], row_height=13)
    y = paragraph(c, "Checkpoint selection used the lowest validation negative "
                   "beta-ELBO. The official test data were never used to update weights "
                   "or choose the checkpoint.", x, y, style=SMALL)

    y = heading(c, "4  Training behavior", x, y)
    y = loss_plot(c, history, x, y)
    y = paragraph(c, "Figure 1: mean negative beta-ELBO (nats/image) on the training "
                   "and validation partitions; lower is better.", x, y, style=SMALL)
    best_row = min(history, key=lambda r: float(r["val_elbo"]))
    y = paragraph(c, f"The selected checkpoint was epoch {best_row['epoch']}, with "
                   f"validation loss {float(best_row['val_elbo']):.1f} "
                   f"(BCE {float(best_row['val_recon']):.1f}, "
                   f"KL {float(best_row['val_kl']):.1f}) nats/image. "
                   f"The final training loss was {float(history[-1]['train_elbo']):.1f}.", x, y)
    if y < 69:
        raise RuntimeError(f"Page 1 content overflowed: y={y:.1f}")
    footer(c, 1)
    c.showPage()

    # Page 2: evaluation, paired examples, discussion, and references.
    x = MARGIN
    y = 718
    y = heading(c, "5  Generative evaluation", x, y)
    y = paragraph(c, f"We generated {evaluation['num_generated']} independent prior "
                   f"samples and compared them with {evaluation['num_real_test']} randomly "
                   "selected official test images. For both metrics, grayscale images were "
                   "copied into three channels, resized to 299 x 299, and processed by "
                   "ImageNet-pretrained Inception-v3 through PyTorch-Ignite [3].", x, y)
    y = draw_table(c, [
        ("Metric", "Measured result"),
        ("Inception Score", f"{evaluation['inception_score']:.3f}"),
        ("FID-1000", f"{evaluation['fid_1000_ignite']:.3f}"),
    ], x, y, [190, TEXT_W - 190], row_height=13)
    y = paragraph(c, "FID-1000 uses the 1000 ImageNet class probabilities as features, "
                   "matching the Ignite tutorial. It is not directly comparable to "
                   "pool3/2048 FID. IS and FID are sensitive to sample count and "
                   "natural-image classifier bias; neither measures diagnostic "
                   "validity [3,4].", x, y, style=SMALL)
    y = heading(c, "6  Qualitative results and discussion", x, y)
    fig_y = y - 2
    fig_left = figure(c, root / "generated_grid.png", x, fig_y,
                      "Figure 2: generated samples", side=158)
    fig_right = figure(c, root / "real_grid.png", x + COL_W + GAP, fig_y,
                       "Figure 3: held-out test images", side=158)
    y = min(fig_left, fig_right)
    y = paragraph(c, "Both grids use the same 64 x 64 grayscale scale. The generated "
                       "samples recover coarse chest shape but lose much of the rib "
                       "and lung detail visible in held-out scans. Generation was "
                       "unconditional; pathology cannot be selected or inferred.", x, y)
    y = heading(c, "7  Limitations", x, y)
    y = paragraph(c, "The 64 x 64 resolution discards fine lung detail. Pixelwise "
                       "BCE can favor smooth averages, while the unit Gaussian prior "
                       "and a single decoder may not capture all acquisition and "
                       "anatomical variation. The training set is class-imbalanced, "
                       "and this unconditional VAE does not control pathology.", x, y)
    y = paragraph(c, "The reported scores are descriptive for this exact "
                        f"{evaluation['num_generated']}-sample experiment. Inception-v3 "
                        "was trained on natural photographs, so its class probabilities "
                        "are a weak proxy for radiographic structure. FID-1000's "
                        "feature choice should accompany every comparison. A clinician "
                        "review or a validated radiology-specific feature model would "
                        "be needed for stronger medical conclusions.", x, y)
    y = heading(c, "8  Conclusion", x, y)
    y = paragraph(c, "A compact convolutional VAE can be trained end to end on "
                        "this dataset and sampled from its learned prior. The loss curve, "
                        "paired examples, IS, and FID-1000 provide a reproducible "
                        "baseline. Further work should test higher resolution, "
                        "conditional generation, and radiology-aware evaluation.", x, y)
    refs_y = heading(c, "References", x, y - 2)
    refs = [
        "[1] P. Mooney. Chest X-Ray Images (Pneumonia). Kaggle, 2018. "
        "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        "[2] D. P. Kingma and M. Welling. Auto-Encoding Variational Bayes. ICLR, 2014.",
        "[3] PyTorch-Ignite. GAN evaluation using FID and IS. "
        "https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/",
        "[4] M. Heusel et al. GANs Trained by a Two Time-Scale Update Rule "
        "Converge to a Local Nash Equilibrium. NeurIPS, 2017.",
    ]
    for ref in refs:
        refs_y = paragraph(c, ref, x, refs_y, TEXT_W, SMALL)
    if refs_y < 69:
        raise RuntimeError(f"Page 2 content overflowed: y={refs_y:.1f}")
    footer(c, 2)
    c.save()
    print(out.resolve())


if __name__ == "__main__":
    main()
