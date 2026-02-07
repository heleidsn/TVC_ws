# ROS坐标系标准说明

## ROS标准坐标系约定 (REP 103)

根据ROS的REP 103（Robot Enhancement Proposal 103），ROS使用以下标准坐标系约定：

### 标准坐标系：FLU (Forward-Left-Up)

**轴定义：**
- **X轴**：向前（Forward）- 机器人前进方向
- **Y轴**：向左（Left）- 机器人左侧方向
- **Z轴**：向上（Up）- 垂直向上

**特点：**
- ✅ **Z轴向上**：符合常见的3D图形和机器人学约定
- ✅ **右手坐标系**：符合右手定则
- ✅ **直观理解**：正值Z表示高度，符合直觉

### 常见坐标系类型

#### 1. **ENU (East-North-Up)** - 地理坐标系
- **X轴**：东（East）
- **Y轴**：北（North）
- **Z轴**：上（Up）
- **用途**：GPS、导航、地理定位

#### 2. **NED (North-East-Down)** - 航空航天坐标系
- **X轴**：北（North）
- **Y轴**：东（East）
- **Z轴**：下（Down）
- **用途**：PX4、无人机、飞行器控制
- **注意**：Z轴向下，与ROS标准相反

#### 3. **FRD (Forward-Right-Down)** - 机体坐标系
- **X轴**：前（Forward）
- **Y轴**：右（Right）
- **Z轴**：下（Down）
- **用途**：飞行器机体坐标系
- **注意**：Z轴向下，与ROS标准相反

## ROS中的坐标系框架（Frames）

### 标准框架命名

1. **`map`** - 全局固定坐标系
   - 通常使用ENU或NED
   - 原点固定在地球上

2. **`odom`** - 里程计坐标系
   - 相对于起始位置的坐标系
   - 通常使用ENU

3. **`base_link`** - 机器人本体坐标系
   - 固定在机器人上
   - 通常使用FLU（X前、Y左、Z上）

4. **`base_footprint`** - 机器人足迹坐标系
   - 机器人在地面上的投影
   - Z轴通常为0（地面高度）

## 本项目中的坐标系使用

### 当前项目（controller_comparison）

**当前使用：NED坐标系**
- X轴：北（North）
- Y轴：东（East）
- Z轴：下（Down）
- **问题**：Z轴向下，与ROS标准（Z轴向上）不一致

### PX4集成部分

根据`README.md`和PX4消息定义：
- **NED Frame**：用于位置和线速度
- **FRD Frame**：用于角速度和机体坐标系

### ROS 2标准

ROS 2遵循REP 103，推荐使用：
- **ENU**用于地理坐标系
- **FLU**用于机器人本体坐标系
- **Z轴向上**作为标准

## 坐标系转换

### NED → ENU转换

```python
# NED to ENU conversion
def ned_to_enu(ned_vector):
    """
    Convert NED (North-East-Down) to ENU (East-North-Up)
    
    Args:
        ned_vector: [x_ned, y_ned, z_ned]
    
    Returns:
        enu_vector: [x_enu, y_enu, z_enu]
    """
    x_ned, y_ned, z_ned = ned_vector
    x_enu = y_ned   # East = East (same)
    y_enu = x_ned   # North = North (same)
    z_enu = -z_ned  # Up = -Down (invert)
    return [x_enu, y_enu, z_enu]
```

### ENU → NED转换

```python
# ENU to NED conversion
def enu_to_ned(enu_vector):
    """
    Convert ENU (East-North-Up) to NED (North-East-Down)
    
    Args:
        enu_vector: [x_enu, y_enu, z_enu]
    
    Returns:
        ned_vector: [x_ned, y_ned, z_ned]
    """
    x_enu, y_enu, z_enu = enu_vector
    x_ned = y_enu   # North = North (same)
    y_ned = x_enu   # East = East (same)
    z_ned = -z_enu  # Down = -Up (invert)
    return [x_ned, y_ned, z_ned]
```

## 建议

### 对于独立仿真系统（controller_comparison）

**推荐：使用ENU坐标系（Z轴向上）**

**理由：**
1. ✅ **符合ROS标准**：与ROS 2的REP 103一致
2. ✅ **更直观**：Z值直接表示高度
3. ✅ **易于理解**：符合大多数用户的期望
4. ✅ **便于集成**：如果未来需要与ROS 2集成，无需转换

### 对于PX4集成部分

**保持：NED坐标系**

**理由：**
1. ✅ **PX4标准**：PX4使用NED作为标准
2. ✅ **兼容性**：与PX4消息格式一致
3. ✅ **行业标准**：航空航天领域广泛使用

## 总结

| 坐标系 | X轴 | Y轴 | Z轴 | 用途 | ROS兼容性 |
|--------|-----|-----|-----|------|----------|
| **FLU** | 前 | 左 | 上 | ROS标准机器人坐标系 | ✅ 完全兼容 |
| **ENU** | 东 | 北 | 上 | ROS地理坐标系 | ✅ 完全兼容 |
| **NED** | 北 | 东 | 下 | 航空航天标准 | ⚠️ Z轴方向相反 |
| **FRD** | 前 | 右 | 下 | 飞行器机体坐标系 | ⚠️ Z轴方向相反 |

**关键点：**
- ROS标准是**Z轴向上**
- NED和FRD是**Z轴向下**（与ROS标准相反）
- 对于独立仿真，建议使用**ENU（Z轴向上）**
- 对于PX4集成，需要保持**NED（Z轴向下）**
