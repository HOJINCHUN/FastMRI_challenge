import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- 경로 설정 ---
recon_dir = "/root/result/test_Varnet/reconstructions_leaderboard/acc8"       # 재구성 결과 폴더
gt_dir    = "/root/Data/samples/acc8/image"                  # GT 폴더
ds_name_1 = "reconstruction"                                     # 재구성 데이터셋 키
ds_name_2 = "image_grappa"                                       # GT 데이터셋 키
ds_name_3 = "image_label"                                        # Alias free 데이터셋 키
out_dir   = '/root/result/test_Varnet/sample_results/acc8'                                           # PNG 저장 폴더

os.makedirs(out_dir, exist_ok=True)

def minmax_normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """이미지 값을 0~1로 정규화 (슬라이스별)"""
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin < eps:
        return np.zeros_like(img, dtype=np.float32)
    return (img - vmin) / (vmax - vmin)

def save_stack_to_pngs(stack: np.ndarray, out_dir: str, prefix: str, step: int = 7):
    """stack: (num_slices, H, W) -> 각 슬라이스를 PNG로 저장"""
    n = stack.shape[0]
    for i in range(0, n, step):
        img = minmax_normalize(stack[i])
        plt.imsave(os.path.join(out_dir, f"{prefix}_slice_{i:03d}.png"), img, cmap="gray")
    if n > 0:
        img0 = minmax_normalize(stack[0])
        plt.imsave(os.path.join(out_dir, f"{prefix}_slice_first.png"), img0, cmap="gray")

# --- 모든 파일 처리 ---
recon_files = sorted([f for f in os.listdir(recon_dir) if f.endswith(".h5")])

for fname in recon_files:
    recon_path = os.path.join(recon_dir, fname)
    print(f"\n[Processing] {fname}")

    # 재구성 결과
    with h5py.File(recon_path, "r") as f:
        recon = np.array(f[ds_name_1])
    print("[Reconstruction] shape:", recon.shape)
    save_stack_to_pngs(recon, out_dir, prefix=f"{fname}_recon")


print("\nDone! Saved PNGs to:", out_dir)


