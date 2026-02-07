# Z-Axis Control Fix for PID Controller

## Problem Analysis

### Rocket Mass and Weight
- **Mass**: 0.6570 kg
- **Weight**: 0.6570 × 9.81 = 6.45 N
- **Base Thrust (for hover)**: 6.45 N

### Coordinate System (NED)
- **Z-axis**: Downward is positive
- **Z < 0**: Above ground (e.g., Z = -1.0 means 1m altitude)
- **Z = 0**: Ground level
- **vz < 0**: Upward velocity (climbing)
- **vz > 0**: Downward velocity (descending)

### Thrust Dynamics

In the rocket dynamics model:
```python
thrust_body[2] = -thrust * cos(phi) * cos(theta)
```

When converted to world frame (NED):
- If attitude is level (phi=0, theta=0): `thrust_world[2] = -thrust` (upward)
- If attitude has tilt: `thrust_world[2] = -thrust * cos(phi) * cos(theta)` (reduced upward component)

Acceleration in NED frame:
```python
accel[2] = (-thrust/mass) - g
```

For upward motion (accel[2] < 0):
- Requires: `thrust > mass * g = 6.45 N`

### Problem Identified

**Issue**: When PID controller generates phi or theta commands (for horizontal control), the vertical thrust component is reduced:
- `thrust_body[2] = -thrust * cos(phi) * cos(theta)`
- If phi=0.2 rad (≈11.5°): `cos(0.2) ≈ 0.98`, so vertical component is reduced by 2%
- If phi=0.3 rad (≈17.2°): `cos(0.3) ≈ 0.96`, so vertical component is reduced by 4%

This causes the rocket to fall even when thrust command is sufficient for hover.

### Solution

**Attitude Compensation**: Compensate for current attitude when calculating thrust:
```python
# Use current attitude from state
current_phi_approx = 2.0 * att[0]  # Approximate roll from qx
current_theta_approx = 2.0 * att[1]  # Approximate pitch from qy

attitude_factor = cos(phi) * cos(theta)
thrust_cmd = base_thrust_cmd / attitude_factor
```

This ensures that the vertical component of thrust remains sufficient even when the rocket tilts.

## Changes Made

1. **Added attitude compensation** in PID controller thrust calculation
2. **Used current attitude** from state (approximated from quaternion)
3. **Limited compensation factor** to prevent division by zero or excessive thrust
4. **Updated default controller selection** to PID only

## Testing

After the fix, the PID controller should:
- Maintain altitude when z_ref < z_current (need to go up)
- Compensate for attitude tilt to maintain vertical thrust component
- Prevent Z-axis from continuously decreasing
