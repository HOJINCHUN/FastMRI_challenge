import h5py, numpy as np

path = "/root/Data/leaderboard/acc8/kspace/brain_test1.h5"
with h5py.File(path, "r") as f:
    ks = f["kspace"][0]        # 한 슬라이스 (coils, H, W)
    m  = np.array(f["mask"]).astype(bool)  # (W,)
    print("mask 평균(샘플링 비율):", m.mean())
    # 에너지로 실제 채워진 라인 비율(마스크와 거의 같아야 함)
    energy_by_pe = np.linalg.norm(ks, axis=tuple(range(ks.ndim-1)))  # (W,)
    print("비제로 라인 비율:", (energy_by_pe > 1e-8).mean())


#결과: yes! it is fully sampled!