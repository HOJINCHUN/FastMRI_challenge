import random
import numpy as np
import torch
from typing import Sequence, Callable, Optional
from scipy.stats import norm
from fastmri.data.subsample import EquispacedMaskFunc
from mraugment.helpers import schedule_p  # schedule_p(D, T, t, p_max, aug_schedule, aug_exp_decay)

# 일반적으로 single coil은 random, multi-coil은 equispace를 더 자주 씀
class BimodalGaussianMaskFunc:
    """
    두 개의 정규분포를 혼합한 가중치로 가속도(R)를 샘플링하는 equispaced 마스크 함수.
    """
    def __init__(
        self,
        center_fractions: Sequence[float],
        min_accel: int,
        max_accel: int,
        mean1: float, stddev1: float,
        mean2: float, stddev2: float,
        mix_weight1: float,
        current_epoch_fn: Callable[[], int],
        args,
    ):
        """
        Args:
            center_fractions: ACS 비율 후보들 (예: [0.08, 0.04])
            min_accel, max_accel: R 최소/최대 (포함)
            mean1,stddev1 / mean2,stddev2: 두 가우시안의 (μ, σ)
            mix_weight1: 첫 번째 가우시안 가중치 (0~1)
            current_epoch_fn: 현재 epoch(int)을 반환하는 콜백
            args: aug_delay, num_epochs, aug_strength, aug_schedule, aug_exp_decay 보유 객체
        """
        if not (0.0 <= mix_weight1 <= 1.0):
            raise ValueError("mix_weight1은 0과 1 사이의 값이어야 합니다.")
        if min_accel > max_accel:
            raise ValueError("min_accel는 max_accel 이하이어야 합니다.")

        self.center_fractions = list(center_fractions)
        self.accelerations = list(range(min_accel, max_accel + 1))

        # 혼합 가중치 계산 (정규화는 random.choices가 내부적으로 해도 되지만 명시적 합리화 OK)
        w1 = np.array([norm.pdf(x, loc=mean1, scale=stddev1) for x in self.accelerations], dtype=np.float64)
        w2 = np.array([norm.pdf(x, loc=mean2, scale=stddev2) for x in self.accelerations], dtype=np.float64)
        w  = mix_weight1 * w1 + (1.0 - mix_weight1) * w2
        w  = np.clip(w, 0.0, None)
        self.weights = (w / (w.sum() + 1e-12)).tolist()

        self.rng = random.Random()
        self.current_epoch_fn = current_epoch_fn
        self.args = args

    def __call__(self, shape: Sequence[int], seed: int, **kwargs) -> torch.Tensor:

        self.rng.seed(seed)

        t = int(self.current_epoch_fn()) if self.current_epoch_fn is not None else 0
        p = schedule_p(
            t,
            D=self.args.aug_delay,
            T=self.args.num_epochs,
            p_max=self.args.aug_strength,
            aug_schedule=self.args.aug_schedule,
            aug_exp_decay=self.args.aug_exp_decay
        )

        # 증강 사용 여부
        use_bimodal = (self.rng.random() < p)

        if use_bimodal:
            # 혼합 분포로 가속도/ACS 샘플
            chosen_accel = self.rng.choices(self.accelerations, weights=self.weights, k=1)[0]
            chosen_cf    = self.rng.choice(self.center_fractions)
        else:
            # 폴백: 파일명 힌트가 있으면 활용, 없으면 기본값(R=8, cf=0.04)
            fname = kwargs.get("fname", "")
            fname_str = str(fname).lower()
            if "acc4" in fname_str:
                chosen_cf, chosen_accel = 0.08, 4
            else:
                chosen_cf, chosen_accel = 0.04, 8

        mask_func = EquispacedMaskFunc(
            center_fractions=[float(chosen_cf)],
            accelerations=[int(chosen_accel)],
        )
        return mask_func(shape, seed)
