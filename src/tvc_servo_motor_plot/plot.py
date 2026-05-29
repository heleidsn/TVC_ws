import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 读取csv文件
# 假设csv文件第一行是表头，包含列名：time, pwm1, pwm2, length1, length2, pitch, roll

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

# 创建子图
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=('Pitch Angle as Function of Lengths', 'Roll Angle as Function of Lengths')
)

# 1. Pitch角度作为Length的二元函数 - 3D曲面
fig.add_trace(
    go.Surface(
        x=X, y=Y, z=Z_pitch,
        colorscale='viridis',
        name='Pitch Angle',
        showscale=True,
        colorbar=dict(x=0.45, title="Pitch angle (deg)")
    ),
    row=1, col=1
)

# 2. Roll角度作为Length的二元函数 - 3D曲面
fig.add_trace(
    go.Surface(
        x=X, y=Y, z=Z_roll,
        colorscale='plasma',
        name='Roll Angle',
        showscale=True,
        colorbar=dict(x=1.0, title="Roll angle (deg)")
    ),
    row=1, col=2
)

# 更新布局
fig.update_layout(
    title_text="3D Surface Plots: Length vs Angle",
    width=1200,
    height=600,
    scene=dict(
        xaxis_title="Pitch servo length (L3, mm)",
        yaxis_title="Roll servo length (L, mm)",
        zaxis_title="Pitch angle (phi, deg)"
    ),
    scene2=dict(
        xaxis_title="Pitch servo length (L3, mm)",
        yaxis_title="Roll servo length (L, mm)",
        zaxis_title="Roll angle (theta, deg)"
    )
)

# 显示图形
fig.show()
