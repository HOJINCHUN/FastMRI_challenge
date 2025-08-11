import h5py
import random
from typing import Optional, Set
from functools import partial
#추가 import for kfold validation
import pandas as pd

from mraugment.data_transforms import VarNetDataTransform
from utils.model.fastmri.data.subsample import MaskFunc
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np


# ---- K-fold helpers ---------------------------------------------------------
def _parse_val_folds(s: Optional[str]) -> Set[int]:
    """예: '2,3' -> {2,3}. None/빈문자면 {1} 기본."""
    if s is None or str(s).strip() == "":
        return {1}
    return {int(x) for x in str(s).replace(" ", "").split(",")}

def _allowed_bases_from_csv(index_csv: str, use_split: str, val_folds_str: Optional[str]) -> Set[str]:
    """
    index_kfold.csv (columns: base,kspace_path,image_path,fold)에서
    - use_split == 'val'  -> fold ∈ val_folds
    - use_split == 'train'-> fold ∉ val_folds
    에 해당하는 base 집합 반환
    """
    df = pd.read_csv(index_csv)
    val_folds = _parse_val_folds(val_folds_str)
    if use_split == "val":
        sub = df[df["fold"].isin(val_folds)]
    else:
        sub = df[~df["fold"].isin(val_folds)]
    bases = set(sub["base"].tolist())    

    if len(bases) == 0:
        raise RuntimeError(f"No samples selected for split={use_split} with val_folds={sorted(val_folds)}")
    return bases
# -----------------------------------------------------------------------------


#기능 1. CPU 병렬화할 때 각 worker의 시드를 다르게 설정. 2. 외부 리소스 초기화
def seed_worker(args, worker_id): 
    worker_seed = args.seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, allowed_bases: Optional[Set[str]] = None):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        self.allowed_bases = allowed_bases

        self.image_examples = []
        self.kspace_examples = []
        
        #isforward가 아니면 image + kspace. isforward가 true면 kspace에서 정보를 가져옴.
        if not forward:
            image_files = sorted((Path(root)/"image").glob("*.h5"))
            for fname in image_files:
                base=fname.stem
                if allowed_bases is not None and base not in allowed_bases:
                    continue
                num_slices = self._get_metadata(fname)
            
                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = sorted((Path(root)/"kspace").glob("*.h5"))
        for fname in kspace_files:
            base = fname.stem
            if allowed_bases is not None and base not in allowed_bases:
                continue
            try:
                num_slices = self._get_metadata(fname)
            except Exception as e:
                raise RuntimeError(f"[KSPACE OPEN FAIL] {fname} :: {e}")

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

        # 간단 정합성 체크: forward가 아닐 때 파일별 총 slice 수가 같아야 함
        if not forward and len(self.image_examples) != len(self.kspace_examples):
            raise ValueError(
                f"image/kspace slice counts mismatch after filtering: "
                f"{len(self.image_examples)} vs {len(self.kspace_examples)}. "
                f"파일명(stem) 매칭/폴더 내용 확인 필요."
            )


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
            else:
                raise KeyError(f"{self.input_key} or {self.target_key} not found in {fname}")
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            kspace = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)

        return self.transform(
            kspace,
            mask,
            target,
            attrs,
            kspace_fname.name,
            dataslice,            # this is your slice_num
        )

def create_data_loaders(data_path, args, augmentor = None, mask_func: Optional[MaskFunc] = None, use_seed: bool = True, shuffle=False, isforward=False, *,
    use_split: str = "train",          # "train" or "val"
    ):
    istrain = (shuffle is True) and (isforward is False)
    index_csv=args.index_csv
    
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key     
    else:
        max_key_ = -1
        target_key_ = -1


    allowed_bases = None
    if index_csv is not None:
        allowed_bases = _allowed_bases_from_csv(
            index_csv=index_csv,
            use_split=use_split,
            val_folds_str=args.set_for_val,
        )
    
    data_storage = SliceData(
        root=data_path,
        transform=VarNetDataTransform(istrain, augmentor, mask_func, use_seed),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward, 
        allowed_bases=allowed_bases,
    )
    
    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers = args.num_workers, 
        pin_memory=istrain,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=lambda worker_id: seed_worker(args, worker_id)
    )
    return data_loader
