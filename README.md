# Chest X-ray variational autoencoder

This project implements and evaluates a convolutional VAE for the [Kaggle Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Reproduce

1. Install Python dependencies from `requirements.txt` (matching PyTorch and torchvision wheels for your platform).
2. Download the Kaggle dataset using `curl -L -o data/chest-xray-pneumonia.zip https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia`, then run `python prepare_data.py`. This extracts `data/chest_xray/train/NORMAL`, `data/chest_xray/train/PNEUMONIA`, and `data/chest_xray/test`. The training scripts also find a nested `chest_xray` directory.
3. Train: `python train.py --data data/chest_xray --epochs 20 --batch-size 64 --threads 4 --seed 285`. The best validation checkpoint, loss history, and a prior-sample grid are written to `results/`.
4. Evaluate: `python evaluate.py --data data/chest_xray --samples 256`. This writes `results/evaluation.json`, plus generated and real sample grids. Pretrained Inception-v3 weights are downloaded by torchvision to `data/torch/hub/checkpoints` if absent.
5. Install [Tectonic](https://tectonic-typesetting.github.io/) and generate the two-page report with `python make_report.py --results results --output output/pdf/chest_xray_vae_report.pdf`. If the compiler is not on `PATH`, pass `--tectonic /path/to/tectonic`. The source is `chest_xray_vae_report.tex`; `neurips_2025.sty` is the unmodified file from the [official 2025 style archive](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip).

The original `train` split is stratified into training and validation partitions using seed 285; the official `test` split is reserved for FID comparison. Each image is converted to grayscale, resized to 64 × 64, and scaled to [0, 1]. The loss is binary cross entropy summed over pixels plus beta times the analytic KL divergence, reported in nats per image. FID and Inception Score use PyTorch-Ignite metrics and the tutorial's default 1000-class probability features from ImageNet Inception-v3. Consequently, the reported FID is **FID-1000**, not the commonly reported pool3/2048 FID. These metrics are limited for medical imagery because their classifier was trained on natural images.

The dataset and downloaded Inception weights are ignored by Git. The generated images are research examples, not diagnostic images.

## Completed run

On the supplied Kaggle dataset, the 20-epoch CPU run used 4,694 training and 522 validation images; the best validation negative ELBO was 2,388.10 nats/image at epoch 20. With 256 generated and 256 official test images, Ignite gave IS 1.779 and FID-1000 0.177. The generated examples retain a coarse chest outline but lack much of the rib and lung texture in the held-out images. The report is compiled using the official NeurIPS 2025 LaTeX style and is two pages including references.
