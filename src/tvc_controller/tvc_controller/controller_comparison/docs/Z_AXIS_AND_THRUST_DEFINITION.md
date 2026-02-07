# Z轴和推力方向定义

## 坐标系：ENU (East-North-Up)

### Z轴方向定义

- **Z轴**：向上为正（Up）
- **Z > 0**：在地面以上（高度）
  - 例如：Z = 1.0 表示1米高度
  - 例如：Z = 5.0 表示5米高度
- **Z = 0**：地面高度
- **Z < 0**：地面以下（无物理意义，飞行中不会出现）

### 速度方向

- **vz > 0**：向上速度（上升）
- **vz = 0**：无垂直速度（悬停）
- **vz < 0**：向下速度（下降）

### 加速度方向

- **az > 0**：向上加速度（上升）
- **az = 0**：无垂直加速度（悬停）
- **az < 0**：向下加速度（下降）

## 重力方向

在ENU坐标系中，重力总是向下（负Z方向）：

```python
gravity_world = np.array([0.0, 0.0, -g])
```

其中 `g = 9.81 m/s²`。

**物理意义**：重力总是向下拉，在ENU坐标系中表示为负Z方向。

## 推力方向

### Body Frame（机体坐标系）

推力沿body +Z轴方向（向上）：

```python
thrust_body = np.array([
    thrust * np.sin(theta),           # body X component
    -thrust * np.sin(phi),            # body Y component
    thrust * np.cos(phi) * np.cos(theta)  # body Z component (向上)
])
```

**关键点**：
- `thrust_body[2] = +thrust * cos(phi) * cos(theta)`（正号，向上）
- 当 `phi = 0, theta = 0` 时，`thrust_body = [0, 0, thrust]`（完全向上）
- 当 `phi` 或 `theta` 不为0时，垂直分量减小，产生水平分量

### World Frame（世界坐标系）

通过旋转矩阵转换到世界坐标系：

```python
thrust_world = R_bw @ thrust_body
```

**关键点**：
- 当姿态水平时，`thrust_world[2] = +thrust`（向上）
- 当姿态倾斜时，`thrust_world[2] = +thrust * cos(phi) * cos(theta)`（垂直分量）

## 加速度计算

总加速度由推力和重力共同决定：

```python
accel_world = (thrust_world / mass) + gravity_world
```

在Z轴方向：

```
accel_world[2] = (thrust_world[2] / mass) - g
```

### 不同推力情况

1. **thrust = mg（平衡推力）**：
   - `accel_world[2] = (mg / m) - g = g - g = 0`
   - **结果**：悬停（无垂直加速度）

2. **thrust > mg（大于平衡推力）**：
   - `accel_world[2] > 0`
   - **结果**：向上加速（上升）

3. **thrust < mg（小于平衡推力）**：
   - `accel_world[2] < 0`
   - **结果**：向下加速（下降）

4. **thrust = 0（无推力）**：
   - `accel_world[2] = 0 - g = -g`
   - **结果**：自由落体（向下加速）

## 平衡推力

平衡推力用于平衡重力，使火箭悬停：

```python
u_eq_thrust = mass * g
```

**示例**（质量 = 0.6570 kg）：
- `u_eq_thrust = 0.6570 × 9.81 = 6.4452 N`

### 控制器中的平衡推力

#### LQR控制器

```python
# 平衡控制输入
u_eq = np.array([0.0, 0.0, mass * g, 0.0])

# LQR控制律
u = u_eq - K * error
```

#### PID控制器

```python
# 基础推力（平衡重力）
thrust_base = mass * g

# 根据误差调整
thrust_cmd = thrust_base + (pos_error_contribution + vel_error_contribution) * mass
```

## 关键验证

### 验证1：悬停条件

当 `thrust = mg` 且姿态水平时：
- `thrust_world[2] = mg`
- `accel_world[2] = (mg / m) - g = 0`
- **结果**：应该悬停（Z轴加速度为0）

### 验证2：上升条件

当 `thrust > mg` 且姿态水平时：
- `thrust_world[2] > mg`
- `accel_world[2] > 0`
- **结果**：应该向上加速

### 验证3：下降条件

当 `thrust < mg` 且姿态水平时：
- `thrust_world[2] < mg`
- `accel_world[2] < 0`
- **结果**：应该向下加速

## 注意事项

1. **符号一致性**：
   - Z轴向上为正
   - 推力向上为正（`thrust_body[2] > 0`）
   - 重力向下为负（`gravity_world[2] < 0`）

2. **平衡点**：
   - 所有控制器都应该在平衡推力（mg）基础上调整
   - LQR控制律：`u = u_eq - K * error`
   - PID控制律：`thrust_cmd = thrust_base + adjustment`

3. **姿态影响**：
   - 当姿态倾斜时，垂直推力分量减小
   - 需要补偿：`thrust_compensated = thrust / (cos(phi) * cos(theta))`

4. **单位一致性**：
   - 位置：米（m）
   - 速度：米/秒（m/s）
   - 加速度：米/秒²（m/s²）
   - 推力：牛顿（N）
   - 质量：千克（kg）
   - 重力加速度：9.81 m/s²
