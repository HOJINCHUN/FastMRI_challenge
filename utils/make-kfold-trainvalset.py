#!/usr/bin/env python3
import argparse
import csv
import json
import random
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional, List, Tuple

ALLOWED_IMAGE_EXTS = [".h5", ".npy", ".pt", ".pth", ".png", ".jpg", ".jpeg", ".tiff"]

def find_matching_image(image_dir: Path, base: str) -> Optional[Path]:
    """Find a file in image_dir whose stem equals `base`."""
    # 1) 우선순위 확장자 탐색
    for ext in ALLOWED_IMAGE_EXTS:
        p = image_dir / f"{base}{ext}"
        if p.exists():
            return p
    # 2) 폴백: 같은 이름의 임의 확장자
    cands = list(image_dir.glob(f"{base}.*"))
    if not cands:
        return None
    if len(cands) > 1:
        exts = [c.suffix for c in cands]
        raise RuntimeError(f"Multiple image files found for base '{base}': {exts}")
    return cands[0]

def main():
    ap = argparse.ArgumentParser(
        description="Build K-fold manifest CSV without moving files (kspace-image paired by stem)."
    )
    ap.add_argument("--root", type=Path, default='/root/Data/trainval',
                    help="Root dir that contains {kspace,image} subdirs. ")
    ap.add_argument("--pattern",       type=str, default="*.h5",
                    help="Filename pattern for kspace volumes (default: *.h5)")
    ap.add_argument("--k",    type=int, default=5)
    ap.add_argument("--seed", type=int, default=430)
    ap.add_argument("--outdir", type=Path, default=Path("/root/Data/trainval"))
    ap.add_argument("--csv_name", type=str, default="index_kfold.csv")
    args = ap.parse_args()

    kspace_dir = args.root /'kspace'
    image_dir  = args.root /'image'
    if not kspace_dir.exists():
        raise SystemExit(f"K-space dir not found: {kspace_dir}")
    if not image_dir.exists():
        raise SystemExit(f"Image dir not found:  {image_dir}")

    # 1) kspace 수집
    kspace_paths: List[Path] = sorted([Path(p) for p in glob(str(kspace_dir / args.pattern))])
    if len(kspace_paths) == 0:
        raise SystemExit(f"No files matched at {kspace_dir} with pattern {args.pattern}")

    # 2) (base, kspace, image) 매칭
    records: List[Tuple[str, Path, Path]] = []
    missing_images: List[str] = []
    for kp in kspace_paths:
        base = kp.stem
        ip = find_matching_image(image_dir, base)
        if ip is None:
            missing_images.append(base)
        else:
            records.append((base, kp, ip))
    if missing_images:
        raise SystemExit(
            f"{len(missing_images)} kspace files have no matching image in {image_dir}.\n"
            f"Examples: {missing_images[:5]}"
        )

    # 3) 재현 가능한 셔플
    rnd = random.Random(args.seed)
    rnd.shuffle(records)

    # 4) K개로 균등 분할해 fold 배정 (1..K)
    N, K = len(records), args.k
    fold_sizes = [N // K + (1 if i < (N % K) else 0) for i in range(K)]
    idx = 0
    fold_of = [None] * N
    for fold_id, fs in enumerate(fold_sizes, start=1):
        for _ in range(fs):
            fold_of[idx] = fold_id
            idx += 1

    # 5) CSV 저장
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / args.csv_name
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base", "kspace_path", "image_path", "fold"])  # 1..K
        for (base, kp, ip), fd in zip(records, fold_of):
            w.writerow([base, str(kp), str(ip), fd])

    # 6) 메타 정보도 남겨두면 좋아요
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "k": K,
        "root": str(args.root),
        "pattern": args.pattern,
        "counts_per_fold": {i: fold_sizes[i-1] for i in range(1, K+1)},
        #각 fold에 몇 개 sample이 있는지
    }
    with open(args.outdir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved: {csv_path}")
    print("Counts per fold:", meta["counts_per_fold"])

if __name__ == "__main__":
    main()