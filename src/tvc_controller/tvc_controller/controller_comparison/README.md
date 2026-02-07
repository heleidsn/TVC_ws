# Controller Comparison Module

This module provides a framework for comparing different control strategies for the TVC (Thrust Vector Control) rocket:

1. **Full-State LQR Controller**: Controls all 12 states (position, velocity, attitude, angular velocity)
2. **Attitude-Only LQR Controller**: Only controls attitude and angular velocity (6 states)
3. **PID Controller**: Cascaded PID controller for position, velocity, attitude, and angular velocity

## Structure

```
controller_comparison/
├── __init__.py                 # Module initialization
├── rocket_dynamics.py          # Rocket dynamics and kinematics simulation
├── lqr_full_state.py           # Full-state LQR controller
├── lqr_attitude_only.py        # Attitude-only LQR controller
├── pid_controller.py           # PID controller
├── comparison_simulator.py     # Main simulation and comparison framework
├── controller_comparison_gui.py # PyQt5 GUI for interactive comparison
├── run_gui.py                  # Launch script for GUI
└── README.md                   # This file
```

## Usage

### GUI Interface (Recommended)

The easiest way to use this module is through the graphical interface:

```bash
cd /home/helei/Documents/TVC_ws/src/tvc_controller/tvc_controller/controller_comparison
python3 run_gui.py
```

Or directly:

```bash
python3 controller_comparison_gui.py
```

**GUI Features:**
- Select controllers to compare (checkboxes)
- Adjust simulation parameters (time, time step)
- Set initial and reference states
- View results in multiple tabs:
  - State trajectories
  - Control inputs
  - Tracking errors
  - 3D trajectory visualization
- View performance metrics in a table

**GUI Layout:**
- **Left Panel**: Controller selection, simulation parameters, initial/reference states, run button, status log
- **Right Panel**: Results displayed in tabs with plots and metrics table

### Basic Example (Python Script)

```python
from tvc_controller.controller_comparison import (
    PhyParams, ComparisonSimulator
)
import numpy as np

# Define physical parameters
phy_params = PhyParams(
    MASS=0.6570,
    G=9.81,
    I_XX=0.062796,
    I_YY=0.062976,
    I_ZZ=0.001403,
    DIST_COM_2_THRUST=0.5693,
)

# Create simulator
simulator = ComparisonSimulator(phy_params, dt=0.01)

# Define initial and reference states
state0 = np.zeros(12)  # Start at origin
state_ref = np.array([
    0.0, 0.0, -1.0,  # Position: 1m above ground (NED frame)
    0.0, 0.0, 0.0,  # Velocity: zero
    0.0, 0.0, 0.0,  # Attitude: level
    0.0, 0.0, 0.0   # Angular velocity: zero
])

# Compare all controllers
results = simulator.compare_controllers(state0, state_ref, t_end=5.0)

# Plot comparison
simulator.plot_comparison(results)

# Compute and print metrics
metrics = simulator.compute_metrics(results)
for name, m in metrics.items():
    print(f"{name}: RMSE = {m['rmse_total']:.4f}")
```

### Running the Example

```bash
cd /home/helei/Documents/TVC_ws/src/tvc_controller/tvc_controller/controller_comparison
python comparison_simulator.py
```

## State Vector

The state vector has 12 dimensions:

```
[x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
```

- **Position** (x, y, z): Position in NED frame (m)
- **Velocity** (vx, vy, vz): Velocity in NED frame (m/s)
- **Attitude** (qx, qy, qz): Quaternion vector part (dimensionless)
- **Angular Velocity** (p, q, r): Angular velocity in FRD body frame (rad/s)

## Control Input

The control input has 4 dimensions:

```
[phi, theta, thrust, tau_r]
```

- **phi**: Thrust deflection angle about X-axis (roll) (rad)
- **theta**: Thrust deflection angle about Y-axis (pitch) (rad)
- **thrust**: Total thrust force (N)
- **tau_r**: Yaw torque (Nm)

## Controllers

### Full-State LQR

The full-state LQR controller uses all 12 states to compute the optimal control input. It solves the continuous-time Algebraic Riccati Equation (ARE) to find the optimal gain matrix K (4x12).

**Advantages:**
- Optimal control in the LQR sense
- Considers coupling between all states
- Good performance for linearized dynamics

**Disadvantages:**
- Requires full state feedback
- May be sensitive to model uncertainties
- Computationally more expensive

### Attitude-Only LQR

The attitude-only LQR controller only controls the 6 attitude-related states (qx, qy, qz, p, q, r). Position and velocity are not directly controlled.

**Advantages:**
- Simpler than full-state LQR
- Focuses on attitude stabilization
- Can be combined with outer-loop position/velocity controllers

**Disadvantages:**
- Does not directly control position/velocity
- May require additional control loops for trajectory tracking

### PID Controller

The PID controller uses cascaded control loops:
- Outer loop: Position PID → velocity reference
- Middle loop: Velocity PID → attitude reference
- Inner loop: Attitude PID → angular velocity reference
- Inner-most loop: Angular velocity PID → control input

**Advantages:**
- Simple and intuitive
- Easy to tune
- Robust to model uncertainties
- Widely used in practice

**Disadvantages:**
- Not optimal in any formal sense
- May have poor performance for highly coupled systems
- Requires manual tuning

## Performance Metrics

The simulator computes several performance metrics:

- **RMSE**: Root Mean Square Error for each state component
- **Max Error**: Maximum absolute error
- **Control Effort**: Integral of squared control input

## Customization

### Custom Physical Parameters

```python
phy_params = PhyParams(
    MASS=0.8,  # kg
    G=9.81,    # m/s^2
    I_XX=0.08, # kg*m^2
    I_YY=0.08, # kg*m^2
    I_ZZ=0.002,# kg*m^2
    DIST_COM_2_THRUST=0.6,  # m
)
```

### Custom LQR Weights

```python
from tvc_controller.controller_comparison import LQRFullStateController

# Custom Q matrix (12x12)
Q = np.diag([10.0, 10.0, 10.0,  # position weights
             5.0, 5.0, 5.0,      # velocity weights
             2.0, 2.0, 0.5,      # attitude weights
             1.0, 1.0, 0.1])     # angular velocity weights

# Custom R matrix (4x4)
R = np.diag([20.0, 20.0, 2.0, 20.0])  # control effort weights

lqr = LQRFullStateController(phy_params, Q=Q, R=R)
```

### Custom PID Gains

```python
from tvc_controller.controller_comparison import PIDController

pid = PIDController(phy_params)
pid.set_gains({
    'Kp_pos': [2.0, 2.0, 2.0],
    'Kd_pos': [1.0, 1.0, 1.0],
    'Kp_att': [3.0, 3.0, 1.0],
    # ... etc
})
```

## Notes

- All simulations use Python (no ROS2 dependencies for the core simulation)
- The dynamics model is a simplified 6-DOF model suitable for TVC rockets
- Control input constraints can be applied during simulation
- The quaternion representation is normalized to prevent drift
