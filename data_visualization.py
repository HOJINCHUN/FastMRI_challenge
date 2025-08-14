import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- 설정 ---
file_path_1 = "/root/result/test_Varnet/sample_results_acc8/brain_test1.h5"          # 재구성 결과(h5)
file_path_2 = "/root/Data/leaderboard/acc8/image/brain_test1.h5"                # GT/GRAPPA(h5)
ds_name_1   = "reconstruction"                                                  # 재구성 데이터셋 키
ds_name_2   = "image_grappa"                                                    # GT 데이터셋 키
ds_name_3  =  "image_label"
out_dir   = "/root/result/test_Varnet/sample_results_acc8"      # 저장 폴더

os.makedirs(out_dir, exist_ok=True)

def minmax_normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """이미지 값을 0~1로 정규화 (슬라이스별)"""
    img = img.astype(np.float32)
    vmin = img.min()
    vmax = img.max()
    if vmax - vmin < eps:
        return np.zeros_like(img, dtype=np.float32)
    return (img - vmin) / (vmax - vmin)

def save_stack_to_pngs(stack: np.ndarray, out_dir: str, prefix: str):
    """
    stack: (num_slices, H, W)
    각 슬라이스를 PNG로 저장.
    """
    step=5
    n = stack.shape[0]
    for i in range(0,n,step):
        img = minmax_normalize(stack[i])
        # cmap='gray'로 저장
        plt.imsave(os.path.join(out_dir, f"{prefix}_slice_{i:03d}.png"), img, cmap="gray")

    # 첫 슬라이스 별도 복사(빠른 확인용)
    if n > 0:
        img0 = minmax_normalize(stack[0])
        plt.imsave(os.path.join(out_dir, f"{prefix}_slice_first.png"), img0, cmap="gray")

# --- 재구성 결과 ---
with h5py.File(file_path_1, "r") as f:
    recon = np.array(f[ds_name_1])  # (num_slices, H, W)
print("[Reconstruction] shape:", recon.shape)
save_stack_to_pngs(recon, out_dir, prefix="recon")

# --- GT/GRAPPA ---
with h5py.File(file_path_2, "r") as f:
    gt = np.array(f[ds_name_2])     # (num_slices, H, W)
    af = np.array(f[ds_name_3])

print("[Aliase free] shape:", af.shape)
save_stack_to_pngs(af, out_dir, prefix="af")
print("[Ground Truth] shape:", gt.shape)
save_stack_to_pngs(gt, out_dir, prefix="gt")

print("Done! Saved PNGs to:")
print(" -", out_dir)

