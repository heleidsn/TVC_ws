import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取csv文件
csv_data_name = "pwm_to_gimbal angles_20250826_155430.csv"

# 读取 CSV 文件
df = pd.read_csv(csv_data_name)

# 获取唯一的length值
length_pitch_unique = sorted(df["Pitch servo length (L3, mm)"].unique())
length_roll_unique = sorted(df["Roll servo length (L, mm)"].unique())

# 创建网格
X, Y = np.meshgrid(length_pitch_unique, length_roll_unique)

# 重塑数据为网格形式
Z_pitch = np.zeros_like(X)
Z_roll = np.zeros_like(X)

for i, length_pitch in enumerate(length_pitch_unique):
    for j, length_roll in enumerate(length_roll_unique):
        # 找到对应的数据点
        mask = (df["Pitch servo length (L3, mm)"] == length_pitch) & (df["Roll servo length (L, mm)"] == length_roll)
        if mask.any():
            Z_pitch[j, i] = df.loc[mask, "Pitch angle (phi, deg)"].iloc[0]
            Z_roll[j, i] = df.loc[mask, "Roll angle (theta, deg)"].iloc[0]

# 创建2D等高线图作为备用方案
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# 1. Pitch角度作为Length的二元函数 - 等高线图
contour1 = axs[0].contourf(X, Y, Z_pitch, levels=20, cmap='viridis')
axs[0].set_xlabel("Pitch servo length (L3, mm)")
axs[0].set_ylabel("Roll servo length (L, mm)")
axs[0].set_title("Pitch Angle as Function of Lengths")
plt.colorbar(contour1, ax=axs[0], label="Pitch angle (phi, deg)")

# 添加等高线
contour_lines1 = axs[0].contour(X, Y, Z_pitch, levels=10, colors='white', alpha=0.5, linewidths=0.5)
axs[0].clabel(contour_lines1, inline=True, fontsize=8, fmt='%.1f')

# 2. Roll角度作为Length的二元函数 - 等高线图
contour2 = axs[1].contourf(X, Y, Z_roll, levels=20, cmap='plasma')
axs[1].set_xlabel("Pitch servo length (L3, mm)")
axs[1].set_ylabel("Roll servo length (L, mm)")
axs[1].set_title("Roll Angle as Function of Lengths")
plt.colorbar(contour2, ax=axs[1], label="Roll angle (theta, deg)")

# 添加等高线
contour_lines2 = axs[1].contour(X, Y, Z_roll, levels=10, colors='white', alpha=0.5, linewidths=0.5)
axs[1].clabel(contour_lines2, inline=True, fontsize=8, fmt='%.1f')

plt.tight_layout()
plt.show()

