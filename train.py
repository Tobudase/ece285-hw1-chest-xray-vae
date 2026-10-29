"""Train a VAE on the Kaggle chest X-ray pneumonia training split."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.utils import save_image

from data_utils import find_dataset_root, image_folder
from vae import ChestVAE, elbo_components


def split_indices(dataset, fraction, seed):
    """Stratified train/validation split, leaving the official test split untouched."""
    rng = random.Random(seed)
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    train, val = [], []
    for indices in by_class.values():
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * fraction))
        val += indices[:n_val]
        train += indices[n_val:]
    rng.shuffle(train)
    return train, val


def pass_epoch(model, loader, optimizer, device, beta):
    """Run one training or validation pass and return per-image mean losses."""
    train = optimizer is not None
    model.train(train)
    totals = torch.zeros(3, dtype=torch.float64)
    seen = 0
    with torch.set_grad_enabled(train):
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            losses = elbo_components(recon, x, mu, logvar, beta)
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses[2].backward()
                optimizer.step()
            totals += torch.tensor([float(v.detach()) for v in losses]) * x.size(0)
            seen += x.size(0)
    return (totals / seen).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/chest_xray")
    p.add_argument("--output", default="results")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=285)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--no-cache", action="store_true", help="decode JPEGs each epoch instead of caching resized tensors")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.val_fraction < 1:
        p.error("epochs/batch-size must be positive and val-fraction must be in (0,1)")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    root = find_dataset_root(args.data)
    output = Path(args.output)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    dataset = image_folder(root, "train")
    train_ids, val_ids = split_indices(dataset, args.val_fraction, args.seed)
    if not args.no_cache:
        # Resize each image once instead of decoding it again every epoch.
        print(f"decoding {len(dataset)} images into an 85 MB in-memory cache", flush=True)
        images = torch.empty((len(dataset), 1, 64, 64), dtype=torch.float32)
        labels = torch.empty(len(dataset), dtype=torch.long)
        for i in range(len(dataset)):
            images[i], labels[i] = dataset[i]
            if (i + 1) % 1000 == 0:
                print(f"decoded {i + 1}/{len(dataset)}", flush=True)
        dataset = TensorDataset(images, labels)
    train_loader = DataLoader(Subset(dataset, train_ids), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_ids), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    model = ChestVAE(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Reuse latent draws so sample grids are comparable across epochs.
    fixed_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    fixed_z = torch.randn(16, args.latent_dim, generator=fixed_generator, device=device)
    history = []
    best = float("inf")
    start = time.time()
    print(f"data={root} train={len(train_ids)} val={len(val_ids)} device={device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        tr = pass_epoch(model, train_loader, optimizer, device, args.beta)
        va = pass_epoch(model, val_loader, None, device, args.beta)
        row = dict(epoch=epoch, train_recon=tr[0], train_kl=tr[1], train_elbo=tr[2],
                   val_recon=va[0], val_kl=va[1], val_elbo=va[2])
        history.append(row)
        with (output / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        # Select the checkpoint using validation loss only.
        if va[2] < best:
            best = va[2]
            torch.save({"model": model.state_dict(), "latent_dim": args.latent_dim,
                        "epoch": epoch, "val_elbo": best, "seed": args.seed},
                       output / "checkpoints" / "best.pt")
        model.eval()
        with torch.no_grad():
            samples = model.decode(fixed_z).cpu()
        save_image(samples, output / "samples.png", nrow=4, padding=3, pad_value=1)
        print(f"epoch {epoch:02d}: train {tr[2]:.2f} (BCE {tr[0]:.2f}, KL {tr[1]:.2f}) "
              f"val {va[2]:.2f} (BCE {va[0]:.2f}, KL {va[1]:.2f})", flush=True)
    info = vars(args) | {"dataset_root": str(root), "train_count": len(train_ids),
                          "val_count": len(val_ids), "device_used": str(device),
                          "best_val_elbo": best, "elapsed_seconds": time.time() - start,
                          "parameter_count": sum(p.numel() for p in model.parameters())}
    (output / "training.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
