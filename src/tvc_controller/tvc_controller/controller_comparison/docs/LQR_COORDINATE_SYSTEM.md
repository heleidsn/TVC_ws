# LQR控制器坐标系说明

## 坐标系定义

### 原始LQR控制器（lqr_controller_node.py）

**坐标系：NED (North-East-Down)**

从代码中可以明确看到：
- 第17行注释：`NED (North, East, Down) frame is used for position and velocity.`
- 第466-468行注释：`Target x/y/z-position in NED frame [meters]`

**控制律处理**：
```python
thrust_gimbal_z_frame = u_lqr[2] - MASS * G
```
- 从LQR输出中减去平衡推力
- 说明LQR输出`u_lqr[2]`已经包含了平衡推力
- 这种处理方式在NED坐标系中更常见

### Controller Comparison中的LQR控制器（lqr_full_state.py）

**坐标系：ENU (East-North-Up)**

**原因**：
1. 使用`RocketDynamics.linearize()`方法
2. `RocketDynamics`类已改为ENU坐标系（从`rocket_dynamics.py`的注释可以看出）
3. 平衡点定义：`state_eq[2] = 0.0`（Z = 0，地面高度，ENU定义）
4. 控制律：`u[2] = mg + (-K[2, :] @ error)`（在平衡推力基础上加上调整）

**控制律处理**：
```python
u = -self.K @ error
u_eq = np.array([0.0, 0.0, self.params.MASS * self.params.G, 0.0])
u = u + u_eq  # 在平衡推力基础上加上LQR调整
```

## 关键区别

### NED坐标系（原始LQR）
- Z轴向下为正
- Z < 0：在地面以上（例如，Z = -1.0 表示1米高度）
- vz < 0：向上速度
- vz > 0：向下速度
- 控制律：`u_lqr[2]`已经包含平衡推力，需要减去mg得到增量

### ENU坐标系（Controller Comparison中的LQR）
- Z轴向上为正
- Z > 0：在地面以上（例如，Z = 1.0 表示1米高度）
- vz > 0：向上速度
- vz < 0：向下速度
- 控制律：`u[2] = mg + adjustment`（在平衡推力基础上加上调整）

## 验证

### Controller Comparison中的LQR（ENU）

测试场景：Z_ref = 1.0 m, Z_current = 0.0 m

```python
error[2] = 0.0 - 1.0 = -1.0  # 负值，需要上升
u[2] = mg - K[2, 2] * (-1.0) = mg + K[2, 2] > mg  # 推力大于平衡推力，可以上升
```

**结果**：✓ 正确，符合ENU逻辑

### 如果是NED坐标系

测试场景：Z_ref = -1.0 m（1米高度），Z_current = 0.0 m（地面）

```python
error[2] = 0.0 - (-1.0) = +1.0  # 正值
u[2] = mg - K[2, 2] * (+1.0) = mg - K[2, 2] < mg  # 推力小于平衡推力，无法上升
```

**结果**：✗ 错误，不符合上升逻辑

## 结论

1. **原始LQR控制器（lqr_controller_node.py）**：明确使用NED坐标系
2. **Controller Comparison中的LQR控制器（lqr_full_state.py）**：使用ENU坐标系
   - 因为它使用`RocketDynamics.linearize()`
   - 而`RocketDynamics`已改为ENU坐标系
   - 与PID控制器使用相同的坐标系（ENU）

## 注意事项

如果要将Controller Comparison中的LQR控制器改为NED坐标系，需要：
1. 修改`RocketDynamics.linearize()`方法，使其基于NED
2. 修改控制律中的平衡推力处理
3. 修改所有Z轴相关的符号和逻辑

但当前实现使用ENU坐标系是正确的，因为：
- 与PID控制器保持一致
- 更直观（Z直接表示高度）
- 符合ROS标准
