import h5py
import random
from typing import Optional
from functools import partial

from mraugment.data_transforms import VarNetDataTransform
from utils.model.fastmri.data.subsample import MaskFunc
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

#기능 1. CPU 병렬화할 때 각 worker의 시드를 다르게 설정. 2. 외부 리소스 초기화
def seed_worker(args, worker_id): 
    worker_seed = args.seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        #isforward가 아니면 image + kspace. isforward가 true면 kspace에서 정보를 가져옴.
        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
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

def create_data_loaders(data_path, args, augmentor = None, mask_func: Optional[MaskFunc] = None, use_seed: bool = True, shuffle=False, isforward=False):
    istrain=False
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
        if shuffle == True: 
            istrain=True            
    else:
        max_key_ = -1
        target_key_ = -1
    
    data_storage = SliceData(
        root=data_path,
        transform=VarNetDataTransform(istrain, augmentor, mask_func, use_seed),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers = args.num_workers, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        worker_init_fn=lambda worker_id: seed_worker(args, worker_id)
    )
    return data_loader
