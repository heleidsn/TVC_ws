# Attitude Representation in the System

## Summary

**The system uses Quaternion (四元数) to represent attitude in the state vector.**

## State Vector Format

The state vector has 12 dimensions:

```
[x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
```

Where:
- **Position**: `[x, y, z]` - Position in ENU frame (m)
- **Velocity**: `[vx, vy, vz]` - Velocity in ENU frame (m/s)
- **Attitude**: `[qx, qy, qz]` - **Quaternion vector part** (dimensionless)
- **Angular Velocity**: `[p, q, r]` - Angular velocity in FRD body frame (rad/s)

## Quaternion Representation

### Storage Format

The state vector stores only the **vector part** of the quaternion:
- `qx, qy, qz`: Quaternion vector components
- `qw`: Quaternion scalar part (computed, not stored)

### Quaternion Scalar Part Calculation

The scalar part `qw` is computed from the vector part:

```python
qw = sqrt(1 - qx² - qy² - qz²)
```

This ensures the quaternion is normalized: `qw² + qx² + qy² + qz² = 1`

### Complete Quaternion

The full quaternion is: `[qw, qx, qy, qz]` (Hamilton convention)

## Why Quaternion?

### Advantages

1. **No Gimbal Lock**: Quaternions avoid the gimbal lock problem that Euler angles have
2. **Smooth Interpolation**: Better for numerical integration and interpolation
3. **Computational Efficiency**: Faster rotation operations
4. **Singularity-Free**: No singularities like Euler angles at ±90° pitch

### Disadvantages

1. **Less Intuitive**: Harder to visualize than Euler angles
2. **Requires Conversion**: Need to convert to Euler angles for display/understanding

## Usage in Code

### In Dynamics (`rocket_dynamics.py`)

```python
# Extract quaternion vector part from state
q_vec = state[6:9]  # [qx, qy, qz]

# Compute scalar part
qw = np.sqrt(np.clip(1.0 - np.sum(q_vec**2), 0.0, 1.0))

# Convert to rotation matrix
R_bw = quaternion_to_rotation_matrix(q_vec)

# Quaternion kinematics
q_vec_dot = 0.5 * (qw * omega + cross(omega, q_vec))
```

### In Controllers

#### PID Controller

```python
# Extract attitude (quaternion vector part)
att = state[6:9]  # [qx, qy, qz]

# Compute attitude error (quaternion difference)
error_att = att_ref - att  # Simple subtraction for small angles

# For small angles, this approximates Euler angle error
```

#### LQR Controller

```python
# State vector includes quaternion vector part
state = [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]

# Linearization assumes small angles
# For small angles: qx ≈ roll/2, qy ≈ pitch/2
```

### Conversion to Euler Angles

When Euler angles are needed (for display, reference, or control):

```python
from scipy.spatial.transform import Rotation

# From quaternion vector part to Euler angles
q_vec = state[6:9]  # [qx, qy, qz]
qw = np.sqrt(np.clip(1.0 - np.sum(q_vec**2), 0.0, 1.0))
quat = [qw, q_vec[0], q_vec[1], q_vec[2]]  # [w, x, y, z]

# Convert to scipy format [x, y, z, w]
quat_scipy = [q_vec[0], q_vec[1], q_vec[2], qw]

# Convert to Euler angles
rot = Rotation.from_quat(quat_scipy)
euler = rot.as_euler('ZYX', degrees=False)  # [yaw, pitch, roll]

# Reorder to [roll, pitch, yaw]
roll = euler[2]
pitch = euler[1]
yaw = euler[0]
```

### Conversion from Euler Angles to Quaternion

When setting reference attitude (e.g., in `compute_attitude_control_only`):

```python
from scipy.spatial.transform import Rotation

# From Euler angles to quaternion
att_ref_euler = [roll, pitch, yaw]  # in radians
rot_ref = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=False)
quat_ref = rot_ref.as_quat()  # [x, y, z, w]
att_ref = quat_ref[:3]  # [qx, qy, qz] - vector part for state vector
```

## Examples

### Zero Attitude (Level)

```python
state = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Attitude: qx=0, qy=0, qz=0
# qw = sqrt(1 - 0) = 1.0
# Full quaternion: [1.0, 0.0, 0.0, 0.0] (no rotation)
```

### Roll = 10 degrees

```python
# Euler: Roll=10°, Pitch=0°, Yaw=0°
# Quaternion (approximately for small angles):
# qx ≈ sin(roll/2) ≈ sin(5°) ≈ 0.087
# qy ≈ 0
# qz ≈ 0
# qw ≈ cos(roll/2) ≈ cos(5°) ≈ 0.996

state[6:9] = [0.087, 0.0, 0.0]
```

## Key Points

1. **State vector stores quaternion vector part only**: `[qx, qy, qz]`
2. **Scalar part is computed**: `qw = sqrt(1 - qx² - qy² - qz²)`
3. **Quaternion is normalized**: `qw² + qx² + qy² + qz² = 1`
4. **For small angles**: `qx ≈ roll/2`, `qy ≈ pitch/2`, `qz ≈ yaw/2`
5. **Conversion needed for display**: Use `Rotation.from_quat()` to get Euler angles
6. **Error calculation**: For small angles, `error_att = att_ref - att` approximates Euler angle error

## References

- `rocket_dynamics.py`: Lines 36-44 (state vector definition)
- `rocket_dynamics.py`: Lines 99-109 (quaternion extraction and computation)
- `pid_controller.py`: Lines 120, 400 (attitude extraction)
- `comparison_simulator.py`: Lines 38-72 (quaternion to Euler conversion)
