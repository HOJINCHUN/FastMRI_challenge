import numpy as np
import matplotlib.pyplot as plt
import os

# 저장된 로그 파일 경로
log_dir = "/root/result/test_Varnet"  # args.val_loss_dir에 해당
v_file_path = os.path.join(log_dir, "val_loss_log.npy")
t_file_path = os.path.join(log_dir, "train_loss_log.npy")

# 로그 불러오기
val_loss_log = np.load(v_file_path)
train_loss_log = np.load(t_file_path)

# 값 확인 출력
print("Validation Loss Log:")
for epoch, loss in val_loss_log:
    print(f"Epoch {int(epoch)}: Val Loss = {loss:.6f}")

print("\nTrain Loss Log:")
for epoch, loss in train_loss_log:
    print(f"Epoch {int(epoch)}: Train Loss = {loss:.6f}")

# 시각화
plt.figure(figsize=(8, 6))
plt.plot(train_loss_log[:, 0], train_loss_log[:, 1], marker='o', label='Train Loss')
plt.plot(val_loss_log[:, 0], val_loss_log[:, 1], marker='s', label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 그래프 저장
plt.savefig(os.path.join(log_dir, "loss_plot.png"))
plt.show()
