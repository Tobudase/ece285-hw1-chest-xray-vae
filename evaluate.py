"""Generate radiographs and calculate Ignite Inception Score and FID.

FID here follows Ignite's tutorial default: ImageNet Inception-v3's 1000
class probabilities are the features. This is not pool3/2048 FID.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from data_utils import find_dataset_root, image_folder
from vae import ChestVAE


def inception_input(x):
    # Ignite's Inception wrapper uses torchvision's transform_input=True,
    # which expects images in [-1, 1] before its channel-wise transform.
    return F.interpolate(x.repeat(1, 3, 1, 1) * 2 - 1,
                         size=(299, 299), mode="bilinear", align_corners=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/chest_xray")
    p.add_argument("--checkpoint", default="results/checkpoints/best.pt")
    p.add_argument("--output", default="results")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=286)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--torch-home", default="data/torch")
    args = p.parse_args()
    if args.samples < 2:
        p.error("At least two samples are needed for FID")
    os.environ["TORCH_HOME"] = str(Path(args.torch_home).resolve())
    torch.set_num_threads(args.threads)
    from ignite.metrics import FID, InceptionScore
    from ignite.metrics.gan.utils import InceptionModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ChestVAE(checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    root = find_dataset_root(args.data)
    dataset = image_folder(root, "test")
    n = min(args.samples, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), n)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    inception = InceptionModel(return_features=False, device="cpu").eval()
    fid = FID(num_features=1000, feature_extractor=nn.Identity())
    inception_score = InceptionScore(num_features=1000, feature_extractor=nn.Identity())
    fid.reset()
    inception_score.reset()
    generated_preview = []
    real_preview = []
    with torch.no_grad():
        for real, _ in loader:
            generated = model.sample(real.size(0), torch.device("cpu"), generator=generator)
            real_probs = inception(inception_input(real))
            gen_probs = inception(inception_input(generated))
            fid.update((real_probs, gen_probs))
            inception_score.update(gen_probs)
            if len(generated_preview) < 2:
                generated_preview.append(generated)
                real_preview.append(real)
            print(f"evaluated {fid._num_examples}/{n}", flush=True)
    save_image(torch.cat(generated_preview)[:16], output / "generated_grid.png",
               nrow=4, padding=3, pad_value=1)
    save_image(torch.cat(real_preview)[:16], output / "real_grid.png",
               nrow=4, padding=3, pad_value=1)
    results = {"fid_1000_ignite": float(fid.compute()),
               "inception_score": float(inception_score.compute()),
               "num_generated": n, "num_real_test": n,
               "test_total": len(dataset), "seed": args.seed,
               "inception_features": "ImageNet Inception-v3 1000-class probabilities",
               "inception_preprocess": "grayscale repeated to RGB; bilinear 299x299; [-1,1]"}
    (output / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
