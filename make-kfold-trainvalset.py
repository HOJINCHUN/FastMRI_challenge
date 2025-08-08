#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# make-kfold-trainvalset.py
#
# Create K-fold train/val splits by VOLUME (per .h5 file) and mirror the files into:
#
# <outdir>/
#   1/
#     train/{kspace,image}/...
#     val/{kspace,image}/...
#   2/ ...
#   ...
#   K/
#
# Assumes:
# - K-space: <root>/Data/train&val/kspace/*.h5
# - Images:  <root>/Data/train&val/image/<matching base name>.*
#
# Reproducible random split with --seed (default 430).
#
# Example:
#   python make-kfold-trainvalset.py --root /path/to/root \
#     --outdir /path/to/kfold-trainvalset --k 5 --seed 430 --mode copy
#
# Tip: Use --mode symlink to save disk space (Linux/macOS).

import argparse
import json
import os
import random
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional, List, Tuple

ALLOWED_IMAGE_EXTS = [".h5", ".npy", ".pt", ".pth", ".png", ".jpg", ".jpeg", ".tiff"]


def find_matching_image(image_dir: Path, base: str) -> Optional[Path]:
    """Find a file in image_dir whose stem equals `base`."""
    candidates: List[Path] = []
    # Preferred extensions first
    for ext in ALLOWED_IMAGE_EXTS:
        p = image_dir / f"{base}{ext}"
        if p.exists():
            candidates.append(p)

    if len(candidates) == 0:
        # Fallback: any extension
        for p in image_dir.glob(f"{base}.*"):
            candidates.append(p)
        # De-duplicate while preserving order
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                unique.append(c)
                seen.add(c)
        candidates = unique

    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        exts = [c.suffix for c in candidates]
        raise RuntimeError(f"Multiple image files found for base '{base}': {exts}")
    return candidates[0]


def copy_or_link(src: Path, dst: Path, mode: str = "copy"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode} (use 'copy' or 'symlink')")


def main():
    parser = argparse.ArgumentParser(description="Create K-fold train/val directory splits (volume-level).")
    parser.add_argument("--root", type=Path,default='root/Data/train&val',
                        help="Project root. Expects Data/train&val/{kspace,image} inside.")
    parser.add_argument("--outdir", type=Path, default='kfold-trainvalset',
                        help="Output directory to create K-fold structure in (e.g., kfold-trainvalset).")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5).")
    parser.add_argument("--seed", type=int, default=430,
                        help="Random seed for reproducible splits (default: 430).")
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "symlink"],
                        help="How to materialize files (default: copy).")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not write/copy any files.")
    parser.add_argument("--kspace-subdir", type=str, default="kspace",
                        help="Relative subdir for kspace files.")
    parser.add_argument("--image-subdir", type=str, default="image",
                        help="Relative subdir for image files.")
    parser.add_argument("--pattern", type=str, default="*.h5",
                        help="Filename pattern for kspace volumes (default: *.h5).")

    args = parser.parse_args()

    kspace_dir = args.root / args.kspace-subdir
    image_dir = args.root / args.image-subdir
    if not kspace_dir.exists():
        raise SystemExit(f"K-space dir not found: {kspace_dir}")
    if not image_dir.exists():
        raise SystemExit(f"Image dir not found:  {image_dir}")

    # 1) Gather kspace volumes
    kspace_paths: List[Path] = sorted([Path(p) for p in glob(str(kspace_dir / args.pattern))])
    if len(kspace_paths) == 0:
        raise SystemExit(f"No files matched at {kspace_dir} with pattern {args.pattern}")

    # 2) Build (base, kspace, image) records
    records: List[Tuple[str, Path, Path]] = []
    missing_images: List[str] = []
    for kp in kspace_paths:
        base = kp.stem  # filename without extension
        ip = find_matching_image(image_dir, base)
        if ip is None:
            missing_images.append(base)
        else:
            records.append((base, kp, ip))

    if missing_images:
        msg = (f"{len(missing_images)} kspace files have no matching image in {image_dir}.\n"
               f"Examples: {missing_images[:5]}")
        raise SystemExit(msg)

    # 3) Reproducible shuffle
    rnd = random.Random(args.seed)
    rnd.shuffle(records)

    # 4) K-fold split indices (balanced chunking of the shuffled list)
    N = len(records)
    K = args.k
    fold_sizes = [N // K + (1 if i < (N % K) else 0) for i in range(K)]
    indices = []
    start = 0
    for fs in fold_sizes:
        indices.append(list(range(start, start + fs)))
        start += fs

    # 5) For each fold: val = this slice; train = others
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Save global mapping to help reproducibility/audit
    mapping = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "k": K,
        "root": str(args.root),
        "kspace_dir": str(kspace_dir),
        "image_dir": str(image_dir),
        "mode": args.mode,
        "files_order_after_shuffle": [r[0] for r in records],  # base names
    }
    if not args.dry_run:
        with open(args.outdir / "folds_global_mapping.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

    for fold_idx in range(K):
        val_idx = set(indices[fold_idx])
        train_idx = [i for i in range(N) if i not in val_idx]

        fold_name = str(fold_idx + 1)  # 1..K
        fold_dir = args.outdir / fold_name
        train_k_dir = fold_dir / "train" / "kspace"
        train_i_dir = fold_dir / "train" / "image"
        val_k_dir = fold_dir / "val" / "kspace"
        val_i_dir = fold_dir / "val" / "image"

        train_list: List[str] = []
        val_list: List[str] = []

        for split, idxs in (("train", train_idx), ("val", list(val_idx))):
            for i in idxs:
                base, kp, ip = records[i]
                if split == "train":
                    dst_k = train_k_dir / kp.name
                    dst_i = train_i_dir / ip.name
                else:
                    dst_k = val_k_dir / kp.name
                    dst_i = val_i_dir / ip.name

                if not args.dry_run:
                    copy_or_link(kp, dst_k, mode=args.mode)
                    copy_or_link(ip, dst_i, mode=args.mode)

                (train_list if split == "train" else val_list).append(base)

        # Write per-fold manifest
        manifest = {
            "fold": fold_idx + 1,
            "seed": args.seed,
            "counts": {"train": len(train_list), "val": len(val_list)},
            "train_bases": train_list,
            "val_bases": val_list,
        }
        if not args.dry_run:
            fold_dir.mkdir(parents=True, exist_ok=True)
            with open(fold_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"[Fold {fold_idx+1}/{K}] train={len(train_list)}  val={len(val_list)} -> {fold_dir}")

    print("Done.")


if __name__ == "__main__":
    main()