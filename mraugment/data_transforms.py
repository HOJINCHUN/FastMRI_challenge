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
from fastmri.data.transforms import to_tensor, apply_mask, center_crop
import torch.nn.functional as F

def _center_pad_to(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = max(0, H - h), max(0, W - w)
    top = ph // 2; bottom = ph - top
    left = pw // 2; right = pw - left
    return F.pad(x, (left, right, top, bottom)) 

class VarNetDataTransform:
    """
    Data Transformer for training VarNet models with added MRAugment data augmentation.
    """

    def __init__(self, istrain: bool, isforward: bool, data_path: str, augmentor = None, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
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
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.data_path = data_path
        self.istrain = istrain
        self.isforward = isforward
        
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
        """
        
        kspace = to_tensor(kspace.astype(np.complex64))  # (..., H, W) or (C,H,W)

        # target 유무 안전 처리
        has_target = isinstance(target, np.ndarray)
        if has_target:
            target = to_tensor(target)
            max_value = float(attrs.get("max", 0.0)) if isinstance(attrs, dict) else 0.0
        else:
            target = torch.tensor(0)
            max_value = 0.0

        # augment는 target 있을 때만
        if self.use_augment and has_target and self.augmentor.schedule_p() > 0.0:
            kspace, target = self.augmentor(kspace, target.shape)
        """
        if len(kspace.shape) == 3:
            kspace.unsqueeze_(0)
        assert len(kspace.shape) == 4"""
        
        seed = None if not self.use_seed else tuple(map(ord, fname))
        padding = None  # 필요시 attrs에서 padding_left/right 읽어 설정
        
        if self.istrain:
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed, padding)
            
        elif self.isforward: #reconstruct
            if "acc4" in str(self.data_path):
               cf, ac = [0.08], [4]
            else:
                cf, ac = [0.04], [8]
            vmask_func = EquispacedMaskFunc(cf, ac)

            masked_kspace, mask = apply_mask(
                kspace, vmask_func, seed, padding
            )  
        
        else: 
            fname_str = str(fname).lower()
            if "acc4" in fname_str:
                cf, ac = [0.08], [4]
            else:
                cf, ac = [0.04], [8]
            vmask_func = EquispacedMaskFunc(cf, ac)

            masked_kspace, mask = apply_mask(
                kspace, vmask_func, seed, padding
            ) 
            
            

        return masked_kspace, mask, target, fname, slice_num, max_value

    
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