# Coordinate System Explanation

## ENU (East-North-Up) Coordinate System - ROS Standard

The simulation uses the **ENU (East-North-Up)** coordinate system, which is the ROS standard (REP 103) and more intuitive for most users.

### Axis Definitions

- **X-axis**: East (positive = eastward)
- **Y-axis**: North (positive = northward)  
- **Z-axis**: Up (positive = upward)

### Important Notes for Z-axis

In ENU frame:
- **Z = 0**: Ground level
- **Z > 0**: Above ground (positive Z = altitude)
  - Example: Z = 1.0 means 1 meter above ground
  - Example: Z = 5.0 means 5 meters above ground
- **Z < 0**: Below ground (not physically meaningful for flight)

### Velocity in ENU Frame

- **vz > 0**: Upward velocity (climbing)
- **vz = 0**: Hovering (no vertical motion)
- **vz < 0**: Downward velocity (descending)

### Acceleration in ENU Frame

- **az > 0**: Upward acceleration (climbing)
- **az = 0**: No vertical acceleration (hovering)
- **az < 0**: Downward acceleration (descending)

### Dynamics Model

The acceleration is calculated as:
```
accel = (thrust_world / mass) + gravity_world
```

Where:
- **gravity_world = [0, 0, -g]**: Gravity points downward (negative Z direction)
- **thrust_world**: Thrust vector in world frame (typically upward, positive Z component)
- When **thrust = 0**: accel[2] = -g (downward acceleration, rocket falls)
- When **thrust = mg**: accel[2] = 0 (no acceleration, rocket hovers)
- When **thrust > mg**: accel[2] > 0 (upward acceleration, rocket climbs)

### Example Scenario

If we want to hover at 1 meter altitude:
- Reference position: `z_ref = 1.0` (1m above ground)
- Current position: `z_current = 0.0` (at ground)
- Position error: `error_z = z_ref - z_current = 1.0 - 0.0 = 1.0`
- To correct: Need to go **up** (increase Z, make it more positive)
- Required action: **Increase thrust** to generate upward acceleration

## PID Controller Z-axis Control

### Control Logic

The PID controller uses the following logic for vertical control in ENU frame:

1. **Position error**: `error_pos[2] = z_ref - z_current`
   - Positive error → need to go up → increase thrust
   - Negative error → need to go down → decrease thrust

2. **Velocity error**: `error_vel[2] = vz_ref - vz_current`
   - Positive error → need more upward velocity → increase thrust
   - Negative error → need more downward velocity → decrease thrust

3. **Thrust calculation**:
   ```python
   thrust_cmd = base_thrust + (pos_error_contribution + vel_error_contribution) * mass
   ```
   - When error is positive (need to go up), this gives: `thrust_cmd > base_thrust` ✓
   - When error is negative (need to go down), this gives: `thrust_cmd < base_thrust` ✓

### Attitude Compensation

When the rocket tilts (phi or theta are non-zero), the vertical component of thrust is reduced:
- `thrust_body[2] = +thrust * cos(phi) * cos(theta)`
- The controller compensates by increasing thrust: `thrust_compensated = thrust / (cos(phi) * cos(theta))`

## Advantages of ENU Coordinate System

1. **Intuitive**: Z directly represents altitude (z = 1.0 means 1m high)
2. **ROS Standard**: Compatible with ROS 2 REP 103 convention
3. **Common Convention**: Matches most 3D graphics and simulation tools
4. **Easier to Understand**: Positive values = height, negative values = below ground

## Comparison with NED

| Aspect | NED (North-East-Down) | ENU (East-North-Up) |
|--------|------------------------|---------------------|
| **Z-axis** | Downward is positive | Upward is positive |
| **Altitude** | Z < 0 (e.g., z = -1.0 = 1m) | Z > 0 (e.g., z = 1.0 = 1m) |
| **Upward velocity** | vz < 0 | vz > 0 |
| **ROS Compatibility** | Not standard | Standard (REP 103) |
| **Intuitiveness** | Less intuitive | More intuitive |
| **Usage** | Aerospace standard | ROS, robotics, 3D graphics |
