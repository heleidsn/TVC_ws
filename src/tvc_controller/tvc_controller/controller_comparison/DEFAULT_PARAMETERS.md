# 默认参数修改指南

本文档说明在哪里可以修改各种默认参数。

## 1. 物理参数（Physical Parameters）

物理参数定义了火箭的质量、重力、惯性矩等物理特性。

### 位置1：GUI初始化（推荐用于GUI运行）

**文件**: `controller_comparison_gui.py`

**位置**: `init_simulator()` 方法（第836-843行）

```python
def init_simulator(self):
    """Initialize the simulator with default parameters"""
    phy_params = PhyParams(
        MASS=0.6570,           # 质量 (kg)
        G=9.81,                # 重力加速度 (m/s²)
        I_XX=0.062796,         # X轴转动惯量 (kg·m²)
        I_YY=0.062976,         # Y轴转动惯量 (kg·m²)
        I_ZZ=0.001403,         # Z轴转动惯量 (kg·m²)
        DIST_COM_2_THRUST=0.5693,  # 质心到推力点距离 (m)
    )
```

### 位置2：命令行脚本

**文件**: `comparison_simulator.py`

**位置**: `main()` 函数（第499-506行）

```python
phy_params = PhyParams(
    MASS=0.6570,
    G=9.81,
    I_XX=0.062796,
    I_YY=0.062976,
    I_ZZ=0.001403,
    DIST_COM_2_THRUST=0.5693,
)
```

---

## 2. PID控制器默认增益

### 位置1：PID控制器类定义（代码中硬编码）

**文件**: `pid_controller.py`

**位置**: `PIDController.__init__()` 方法（第42-60行）

```python
# Position PID gains [x, y, z]
self.Kp_pos = np.array([1.0, 1.0, 10.0])
self.Ki_pos = np.array([0.0, 0.0, 0.0])
self.Kd_pos = np.array([0.5, 0.5, 0.5])

# Velocity PID gains [vx, vy, vz]
self.Kp_vel = np.array([1.0, 1.0, 1.0])
self.Ki_vel = np.array([0.0, 0.0, 0.0])
self.Kd_vel = np.array([0.1, 0.1, 0.1])

# Attitude PID gains [qx, qy, qz]
self.Kp_att = np.array([2.0, 2.0, 0.5])
self.Ki_att = np.array([0.0, 0.0, 0.0])
self.Kd_att = np.array([0.5, 0.5, 0.1])

# Angular velocity PID gains [p, q, r]
self.Kp_omega = np.array([1.0, 1.0, 0.5])
self.Ki_omega = np.array([0.0, 0.0, 0.0])
self.Kd_omega = np.array([0.1, 0.1, 0.05])
```

### 位置2：GUI默认值（GUI界面显示）

**文件**: `controller_comparison_gui.py`

**位置**: `create_control_panel()` 方法中的PID参数标签页（第428-592行）

```python
# Position PID gains
default_Kp_pos = [1.0, 1.0, 10.0]
default_Ki_pos = [0.0, 0.0, 0.0]
default_Kd_pos = [0.5, 0.5, 0.5]

# Velocity PID gains
default_Kp_vel = [1.0, 1.0, 1.0]
default_Ki_vel = [0.0, 0.0, 0.0]
default_Kd_vel = [0.1, 0.1, 0.1]

# Attitude PID gains
default_Kp_att = [2.0, 2.0, 0.5]
default_Ki_att = [0.0, 0.0, 0.0]
default_Kd_att = [0.5, 0.5, 0.1]

# Angular velocity PID gains
default_Kp_omega = [1.0, 1.0, 0.5]
default_Ki_omega = [0.0, 0.0, 0.0]
default_Kd_omega = [0.1, 0.1, 0.05]
```

**注意**: GUI中的值会在界面初始化时显示，但实际运行时使用的是GUI输入框中的值。

---

## 3. LQR控制器默认Q/R矩阵

### Full-State LQR

**文件**: `lqr_full_state.py`

**位置**: `LQRFullStateController.__init__()` 方法（第49-60行）

```python
# Default Q matrix (12x12) - 状态权重
Q_diag = np.array([
    1.0, 1.0, 1.0,      # position [x, y, z]
    1.0, 1.0, 1.0,      # velocity [vx, vy, vz]
    1.0, 1.0, 0.1,      # attitude [qx, qy, qz] (lower weight on qz)
    1.0, 1.0, 0.01      # angular velocity [p, q, r] (lower weight on r)
])
self.Q = np.diag(Q_diag)

# Default R matrix (4x4) - 控制权重
R_diag = np.array([10.0, 10.0, 1.0, 10.0])  # [phi, theta, thrust, tau_r]
self.R = np.diag(R_diag)
```

**GUI默认值位置**: `controller_comparison_gui.py` 第358-400行

```python
# Q matrix diagonal elements (12 states)
default_Q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.01]

# R matrix diagonal elements (4 controls)
default_R = [10.0, 10.0, 1.0, 10.0]
```

### Attitude-Only LQR

**文件**: `lqr_attitude_only.py`

**位置**: `LQRAttitudeOnlyController.__init__()` 方法

```python
# Default Q_att matrix (6x6) - 姿态状态权重
Q_att_diag = np.array([1.0, 1.0, 0.1, 1.0, 1.0, 0.01])  # [qx, qy, qz, p, q, r]

# Default R_att matrix (3x3) - 姿态控制权重
R_att_diag = np.array([10.0, 10.0, 10.0])  # [phi, theta, tau_r]
```

**GUI默认值位置**: `controller_comparison_gui.py` 第630-658行

```python
default_Q_att = [1.0, 1.0, 0.1, 1.0, 1.0, 0.01]
default_R_att = [10.0, 10.0, 10.0]
```

---

## 4. 仿真参数

### 时间步长（dt）

**文件**: `comparison_simulator.py`

**位置**: `ComparisonSimulator.__init__()` 方法（第52行）

```python
def __init__(self, phy_params: PhyParams, dt: float = 0.01):
    # dt: 时间步长，默认0.01秒
```

**GUI位置**: `controller_comparison_gui.py` 第760-770行

```python
self.spin_dt = QDoubleSpinBox()
self.spin_dt.setRange(0.001, 0.1)
self.spin_dt.setValue(0.01)  # 默认值
```

### 仿真时长（t_end）

**GUI位置**: `controller_comparison_gui.py` 第750-760行

```python
self.spin_t_end = QDoubleSpinBox()
self.spin_t_end.setRange(0.1, 100.0)
self.spin_t_end.setValue(5.0)  # 默认值：5秒
```

**命令行脚本**: `comparison_simulator.py` 第529行

```python
results = simulator.compare_controllers(state0, state_ref, t_end=5.0)
```

---

## 5. 初始状态和参考状态

### 命令行脚本

**文件**: `comparison_simulator.py`

**位置**: `main()` 函数（第512-525行）

```python
# Initial state
state0 = np.array([
    0.0, 0.0, 0.0,  # position [x, y, z] (m)
    0.0, 0.0, 0.0,  # velocity [vx, vy, vz] (m/s)
    0.0, 0.0, 0.0,  # attitude [qx, qy, qz]
    0.0, 0.0, 0.0   # angular velocity [p, q, r] (rad/s)
])

# Reference state
state_ref = np.array([
    0.0, 0.0, 1.0,  # position [x, y, z] (1m above ground in ENU)
    0.0, 0.0, 0.0,  # velocity
    0.0, 0.0, 0.0,  # attitude
    0.0, 0.0, 0.0   # angular velocity
])
```

### GUI界面

**文件**: `controller_comparison_gui.py`

**位置**: `create_control_panel()` 方法

- **初始状态**: 第697-729行
  - `self.spin_x0`, `self.spin_y0`, `self.spin_z0` (默认: 0.0, 0.0, 0.0)
  - `self.spin_vx0`, `self.spin_vy0`, `self.spin_vz0` (默认: 0.0, 0.0, 0.0)
  - 等等...

- **参考状态**: 第730-767行
  - `self.spin_x_ref`, `self.spin_y_ref`, `self.spin_z_ref` (默认: 0.0, 0.0, 1.0)
  - `self.spin_vx_ref`, `self.spin_vy_ref`, `self.spin_vz_ref` (默认: 0.0, 0.0, 0.0)
  - 等等...

---

## 修改建议

### 对于GUI用户

1. **物理参数**: 修改 `controller_comparison_gui.py` 的 `init_simulator()` 方法
2. **控制器参数**: 直接在GUI界面中调整（无需修改代码）
3. **初始/参考状态**: 直接在GUI界面中调整

### 对于代码用户

1. **物理参数**: 修改 `comparison_simulator.py` 的 `main()` 函数
2. **PID增益**: 修改 `pid_controller.py` 的 `__init__()` 方法，或创建后手动设置
3. **LQR Q/R**: 修改 `lqr_full_state.py` 或 `lqr_attitude_only.py` 的 `__init__()` 方法，或在创建时传入自定义矩阵

### 推荐做法

- **GUI运行**: 使用GUI界面调整参数（最方便）
- **脚本运行**: 在 `comparison_simulator.py` 的 `main()` 函数中修改
- **代码集成**: 在创建控制器时传入自定义参数

---

## 参数说明

### 物理参数

- **MASS**: 火箭质量（kg）
- **G**: 重力加速度（m/s²），通常为9.81
- **I_XX, I_YY, I_ZZ**: 绕X、Y、Z轴的转动惯量（kg·m²）
- **DIST_COM_2_THRUST**: 质心到推力点的距离（m）

### PID增益

- **Kp**: 比例增益，响应速度
- **Ki**: 积分增益，消除稳态误差
- **Kd**: 微分增益，减少超调

### LQR权重

- **Q矩阵**: 状态权重，较大的值表示更重视该状态
- **R矩阵**: 控制权重，较大的值表示更限制控制输入
