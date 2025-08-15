"""
Model dependent data transforms that apply MRAugment to 
training data before fed to the model.
Modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/data/transforms.py
"""
from typing import Dict, Optional, Sequence, Tuple, Union
import fastmri
import numpy as np
import torch

from fastmri.data.subsample import MaskFunc
from fastmri.data.subsample import EquispacedMaskFunc
from fastmri.data.transforms import to_tensor, apply_mask


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models with added MRAugment data augmentation.
    """

    def __init__(self, isforward: bool, istrain: bool, augmentor = None, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            augmentor: DataAugmentor object that encompasses the MRAugment pipeline and
                schedules the augmentation probability
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.istrain=istrain
        self.mask_func = mask_func
        self.use_seed = use_seed
        
        if augmentor is not None:
            self.use_augment = True
            self.augmentor = augmentor
        else:
            self.use_augment = False

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, float, torch.Tensor]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            tuple containing:
                masked_kspace: k-space after applying sampling mask.
                mask: The applied sampling mask
                target: The target image (if applicable).
                fname: File name.
                slice_num: The slice index.
                max_value: Maximum image value.
                crop_size: The size to crop the final image.
        """
        kspace = to_tensor(kspace.astype(np.complex64))  # (..., H, W) or (C,H,W)

        # target 유무 안전 처리
        has_target = isinstance(target, np.ndarray)
        if has_target:
            target = to_tensor(target.astype(np.float32))
            max_value = float(attrs.get("max", 0.0)) if isinstance(attrs, dict) else 0.0
        else:
            target = -1
            max_value = -1

        # augment는 target 있을 때만
        if self.use_augment and has_target and self.augmentor.schedule_p() > 0.0:
            kspace, target = self.augmentor(kspace, target.shape)

        # (C,H,W) → (1,C,H,W)
        if kspace.dim() == 3:
            kspace = kspace.unsqueeze(0)
        assert kspace.dim() == 4  # [B?, C, H, W] per-sample로는 [C,H,W]

        seed = None if not self.use_seed else tuple(map(ord, fname))
        padding = None  # 필요시 attrs에서 padding_left/right 읽어 설정

        if self.mask_func:
            # train/val/test 공통: apply_mask 사용
            masked_kspace, mask_t = apply_mask(kspace, self.mask_func, seed, padding)
            mask_t = (mask_t > 0.5)
        else:
            # HDF5의 1D mask 사용
            W = kspace.shape[-2]
            mask_np = (np.asarray(mask) > 0.5).astype(np.float32).reshape(1, 1, W, 1)
            mask_t = torch.from_numpy(mask_np).to(device=kspace.device)
            masked_kspace = kspace * mask_t.to(kspace.dtype)
            mask_t = mask_t.bool()

        # crop_size: target 없으면 kspace에서 대체
        if has_target:
            crop_size = torch.tensor([target.shape[-2], target.shape[-1]])
        else:
            crop_size = torch.tensor([kspace.shape[-3], kspace.shape[-2]])

        return masked_kspace, mask_t, target, fname, slice_num, max_value, crop_size

    
    def seed_pipeline(self, seed):
        """
        Sets random seed for the MRAugment pipeline. It is important to provide
        different seed to different workers and across different GPUs to keep
        the augmentations diverse.
        
        For an example how to set it see worker_init in pl_modules/fastmri_data_module.py
        """
        if self.use_augment:
            if self.augmentor.aug_on:
                self.augmentor.augmentation_pipeline.rng.seed(seed)