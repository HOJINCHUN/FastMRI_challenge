import random
import numpy as np
import torch
from typing import Sequence
from scipy.stats import norm
from fastmri.data.subsample import EquispacedMaskFunc

class BimodalGaussianMaskFunc:
    """
    두 개의 정규분포를 결합하여 만든 가중치로 가속도를 샘플링하는 클래스.
    두 개의 중심(peak)을 가지는 분포를 만들 수 있습니다.
    """
    def __init__(self,
                 center_fractions: Sequence[float],
                 min_accel: int,
                 max_accel: int,
                 mean1: float, stddev1: float,
                 mean2: float, stddev2: float,
                 mix_weight1: float):
        """
        Args:
            center_fractions, min_accel, max_accel: 이전과 동일.
            mean1, stddev1: 첫 번째 정규분포의 평균과 표준편차 (예: 4.0, 1.0).
            mean2, stddev2: 두 번째 정규분포의 평균과 표준편차 (예: 8.0, 1.5).
            mix_weight1: 첫 번째 정규분포(mean1)에 대한 가중치 (0.0 ~ 1.0).
                         두 번째 분포의 가중치는 (1 - mix_weight1)로 자동 계산됨.
        """
        self.center_fractions = center_fractions
        self.accelerations = list(range(min_accel, max_accel + 1))
        
        # 1. 각 정규분포에 대한 가중치 계산
        weights1 = np.array([norm.pdf(x, loc=mean1, scale=stddev1) for x in self.accelerations])
        weights2 = np.array([norm.pdf(x, loc=mean2, scale=stddev2) for x in self.accelerations])

        # 2. 두 분포의 가중치를 mix_weight1을 이용해 결합
        if not (0.0 <= mix_weight1 <= 1.0):
            raise ValueError("mix_weight1은 0과 1 사이의 값이어야 합니다.")
            
        self.weights = (mix_weight1 * weights1) + ((1 - mix_weight1) * weights2)
        
        self.rng = random.Random()

    def __call__(self, shape: Sequence[int], seed: int, **kwargs) -> torch.Tensor:
        """
        결합된 가중치에 따라 하나의 가속도를 선택하고 마스크를 생성합니다.
        """
        self.rng.seed(seed)

        # 가중치를 기반으로 가속도 무작위 선택
        chosen_accel = self.rng.choices(self.accelerations, weights=self.weights, k=1)[0]
        chosen_cf = self.rng.choice(self.center_fractions)

        # 선택된 단일 옵션으로 표준 EquispacedMaskFunc 인스턴스 생성
        mask_func = EquispacedMaskFunc(
            center_fractions=[chosen_cf],
            accelerations=[chosen_accel]
        )
        return mask_func(shape, seed)