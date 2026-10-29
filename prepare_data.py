"""Extract only the train/test image tree from the Kaggle archive.

The public archive contains duplicate nested copies of chest_xray on some
releases. This selects the shortest complete path and avoids duplicate files.
"""

import argparse
import collections
import zipfile
from pathlib import Path, PurePosixPath


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", nargs="?", default="data/chest-xray-pneumonia.zip")
    p.add_argument("--output", default="data/chest_xray")
    args = p.parse_args()
    out = Path(args.output)
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        prefixes = set()
        # Some releases nest duplicate copies; choose one complete train tree.
        for name in names:
            parts = PurePosixPath(name).parts
            for i in range(len(parts) - 2):
                if parts[i] == "train" and parts[i + 1] in ("NORMAL", "PNEUMONIA"):
                    prefixes.add(parts[:i])
        if not prefixes:
            raise RuntimeError("Could not find a chest_xray/train directory in archive")
        prefix = min(prefixes, key=lambda x: (len(x), x))
        counts = collections.Counter()
        # Extract image files only from the chosen train/test tree.
        for name in names:
            parts = PurePosixPath(name).parts
            if parts[:len(prefix)] != prefix:
                continue
            rel = parts[len(prefix):]
            if len(rel) != 3 or rel[0] not in ("train", "test") or rel[1] not in ("NORMAL", "PNEUMONIA"):
                continue
            if Path(rel[2]).suffix.lower() not in (".jpeg", ".jpg", ".png"):
                continue
            dest = out.joinpath(*rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src, dest.open("wb") as target:
                while block := src.read(1024 * 1024):
                    target.write(block)
            counts[(rel[0], rel[1])] += 1
    print(f"extracted from {'/'.join(prefix)} to {out}")
    for key, value in sorted(counts.items()):
        print(f"{key[0]}/{key[1]}: {value}")


if __name__ == "__main__":
    main()
