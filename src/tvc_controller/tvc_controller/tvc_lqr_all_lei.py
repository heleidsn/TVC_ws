#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LQR Controller for TVC Platform
This script defines the LQRController class and provides functions for testing and simulation.
"""

import numpy as np
import scipy.linalg
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from enum import Enum

@dataclass
class PhyParams:
    MASS: float
    G: float
    I_XX: float
    I_YY: float
    I_ZZ: float
    DIST_COM_2_THRUST: float


class SimulationMode(Enum):
    """Simulation mode enumeration"""
    NO_LIMITATION = "no_limitation"  # No limitations, no actuator dynamics
    ACTUATOR_LIMITATION = "actuator_limitation"  # Only actuator limitations, no actuator dynamics
    POSITION_ERROR_LIMITATION = "position_error_limitation"  # Only position error limitations, no actuator dynamics
    ACTUATOR_DYNAMICS = "actuator_dynamics"  # Only actuator dynamics, no other limitations
    ACTUATOR_LIMITATION_AND_DYNAMICS = "actuator_limitation_and_dynamics"  # Actuator limitations + dynamics
    POSITION_ERROR_LIMITATION_AND_DYNAMICS = "position_error_limitation_and_dynamics"  # Position error limitations + dynamics
    ALL = "all"  # All limitations and dynamics included


class LQRController: 
    def __init__(self, phy_params: PhyParams):
        """
        LQR controller based on linearized TVC platform (consistent with your previous code).
        """
        if phy_params is None:
            raise ValueError("Physical parameters must be provided to initialize the LQR controller.")
        
        # Initialize system properties from physical parameters
        self.mass = phy_params.MASS
        self.gravity = phy_params.G
        self.I_XX = phy_params.I_XX
        self.I_YY = phy_params.I_YY
        self.I_ZZ = phy_params.I_ZZ
        self.dist_com_2_thrust = phy_params.DIST_COM_2_THRUST
        
        # System matrices
        self.A = self.set_A()
        assert self.A.shape == (12, 12), "Matrix A must be 12x12."
        
        self.B = self.set_B()
        assert self.B.shape == (12, 4), "Matrix B must be 12x4."
        
        # LQR weights & solution
        self.Q = None
        self.R = None
        self.P = None
        self.K = None

    # ---------- LQR Solution ----------
    def solve_lqr(self) -> bool:
        """
        Solve continuous-time LQR: A'P + P A - P B R^-1 B' P + Q = 0
        K = R^-1 B' P
        """
        if self.Q is None or self.R is None:
            raise ValueError("Q and R must be set before solving LQR.")
        try:
            self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            # K = R^{-1} B^T P
            self.K = np.linalg.solve(self.R, self.B.T @ self.P)
            return True
        except Exception as e:
            print(f"Error solving LQR: {e}")
            self.K = None
            self.P = None
            return False

    def get_K(self) -> np.ndarray:
        """Return LQR gain K (4x12)"""
        return self.K

    def get_P(self) -> np.ndarray:
        """Return Riccati equation solution P (12x12)"""
        return self.P

    # ---------- System Matrix A ----------
    def set_A(self):
        """
        A matrix for linearized TVC/quadrotor-like model
        State: [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
        """
        g = self.gravity
        A = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0,   0,   0,   0,   0],   # x_dot = vx
            [0, 0, 0, 0, 1, 0, 0, 0,   0,   0,   0,   0],   # y_dot = vy
            [0, 0, 0, 0, 0, 1, 0, 0,   0,   0,   0,   0],   # z_dot = vz
            [0, 0, 0, 0, 0, 0, 0, -2*g, 0,  0,   0,   0],   # vx_dot ≈ -2g qy  (based on your original notation)
            [0, 0, 0, 0, 0, 0, 2*g,  0,  0,  0,   0,   0],   # vy_dot ≈  2g qx
            [0, 0, 0, 0, 0, 0, 0,  0,  0,  0,   0,   0],   # vz_dot, thrust/mass part is in B matrix
            [0, 0, 0, 0, 0, 0, 0,  0,  0,  0.5, 0,   0],   # qx_dot = p/2
            [0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0.5, 0],   # qy_dot = q/2
            [0, 0, 0, 0, 0, 0, 0,  0,  0,  0,   0,  0.5], # qz_dot = r/2
            [0, 0, 0, 0, 0, 0, 0,  0,  0,  0,   0,   0],  # p_dot
            [0, 0, 0, 0, 0, 0, 0,  0,  0,  0,   0,   0],  # q_dot
            [0, 0, 0, 0, 0, 0, 0,  0,  0,  0,   0,   0],  # r_dot
        ])
        return A

    # ---------- System Matrix B ----------
    def set_B(self):
        """
        Control input: u = [qx_cmd, qy_cmd, thrust_cmd, r_cmd]
        Corresponds to your original B matrix formulation.
        """
        g = self.gravity
        m = self.mass
        l = self.dist_com_2_thrust
        Ixx = self.I_XX
        Iyy = self.I_YY
        Izz = self.I_ZZ

        B = np.array([
            [0, 0, 0, 0],  # x
            [0, 0, 0, 0],  # y
            [0, 0, 0, 0],  # z
            [0,    -g, 0, 0],       # vx
            [g,     0, 0, 0],       # vy
            [0,     0, 1/m, 0],     # vz
            [0,     0, 0, 0],       # qx
            [0,     0, 0, 0],       # qy
            [0,     0, 0, 0],       # qz
            [-l*m*g/Ixx, 0, 0, 0],  # p
            [0, -l*m*g/Iyy, 0, 0],  # q
            [0,     0, 0, 1/Izz],   # r
        ])
        return B

    # ---------- Set Q / R ----------
    def set_Q(self, Q: np.ndarray) -> bool:
        if Q is None:
            return False
        Q = np.asarray(Q)
        if Q.shape != (12, 12):
            raise ValueError(f"Q must be 12x12, got {Q.shape}")
        self.Q = Q
        return True

    def set_R(self, R: np.ndarray) -> bool:
        if R is None:
            return False
        R = np.asarray(R)
        if R.shape != (4, 4):
            raise ValueError(f"R must be 4x4, got {R.shape}")
        self.R = R
        return True


# ============================================================================
# Parameter Definition Functions
# ============================================================================

def load_parameters() -> Tuple[PhyParams, np.ndarray, np.ndarray, Dict]:
    """
    Load physical parameters, LQR weights, and control constraints.
    
    Returns:
        phy_params: Physical parameters
        Q_matrix: LQR state weight matrix (12x12)
        R_matrix: LQR control weight matrix (4x4)
        constraints: Dictionary containing control input constraints
    """
    # Physical parameters (from load_parameters default values)
    phy_params = PhyParams(
        MASS=0.6570,
        G=9.81,
        I_XX=0.062796,
        I_YY=0.062976,
        I_ZZ=0.001403,
        DIST_COM_2_THRUST=0.5693,
    )

    # LQR Q, R diagonal (from default_Q / default_R in load_parameters)
    Q_diagonal = np.array([1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0, 1.0, 0.1,
                           1.0, 1.0, 0.01])

    R_diagonal = np.array([10.0, 10.0, 1.0, 10.0])

    Q_matrix = np.diag(Q_diagonal)
    R_matrix = np.diag(R_diagonal)

    constraints = {}

    return phy_params, Q_matrix, R_matrix, constraints


def print_parameters(phy_params: PhyParams, x_ref: np.ndarray, constraints: Dict):
    """Print loaded parameters and constraints."""
    print("x_ref shape:", x_ref.shape)
    print("x_ref:", x_ref)
    print("\nControl Input Constraints:")
    print(f"  qx_cmd (thrust deflection angle): [{constraints['servo_0_min']:.3f}, {constraints['servo_0_max']:.3f}] rad")
    print(f"  qy_cmd (thrust deflection angle): [{constraints['servo_1_min']:.3f}, {constraints['servo_1_max']:.3f}] rad")
    print(f"  Total thrust: [{constraints['thrust_total_min']:.2f}, {constraints['thrust_total_max']:.2f}] N")
    print(f"    (thrust_cmd range: [{constraints['thrust_cmd_min']:.2f}, {constraints['thrust_cmd_max']:.2f}] N)")
    print(f"    (Thrust-to-weight ratio: {constraints['thrust_total_max'] / (phy_params.MASS * phy_params.G):.2f})")
    print(f"  r_cmd (yaw torque): [{constraints['r_cmd_min']:.3f}, {constraints['r_cmd_max']:.3f}] Nm")


# ============================================================================
# LQR Controller Setup
# ============================================================================

def setup_lqr_controller(phy_params: PhyParams, Q_matrix: np.ndarray, 
                         R_matrix: np.ndarray) -> Tuple[LQRController, bool]:
    """
    Create and configure LQR controller.
    
    Returns:
        lqr: Configured LQR controller
        success: Whether LQR solution was successful
    """
    lqr = LQRController(phy_params)
    lqr.set_Q(Q_matrix)
    lqr.set_R(R_matrix)
    
    success = lqr.solve_lqr()
    return lqr, success


def print_lqr_results(lqr: LQRController, success: bool):
    """Print LQR solution results."""
    print("LQR solve success:", success)
    if success:
        K = lqr.get_K()
        print("K shape:", K.shape)
        print("K =\n", K)


# ============================================================================
# Control Test Functions
# ============================================================================

def test_control_law(lqr: LQRController, x_ref: np.ndarray, x0: np.ndarray = None):
    """
    Test control law: compute control input for given initial state.
    
    Args:
        lqr: LQR controller
        x_ref: Reference state
        x0: Initial state (default: zeros)
    
    Returns:
        u: Control input
        e: State error
    """
    if x0 is None:
        x0 = np.zeros(12)
    
    e = x0 - x_ref
    K = lqr.get_K()
    u = -K @ e
    
    print("Initial state x0:", x0)
    print("Reference state x_ref:", x_ref)
    print("State error e = x0 - x_ref:", e)
    print("Control input u = -K e:", u)
    
    return u, e


# ============================================================================
# Error Limiting Functions
# ============================================================================

def limit_state_error(e: np.ndarray, constraints: Dict) -> np.ndarray:
    """
    Limit state error before computing LQR control law.
    
    Args:
        e: State error vector [pos_error(3), vel_error(3), att_error(3), angvel_error(3)]
        constraints: Constraints dictionary containing error limits
    
    Returns:
        e_limited: Limited state error vector
    """
    e_limited = np.zeros_like(e)
    
    # Limit position error
    e_limited[0:3] = np.clip(e[0:3], -constraints['error_pos_max'], constraints['error_pos_max'])
    
    # Limit velocity error
    e_limited[3:6] = np.clip(e[3:6], -constraints['error_vel_max'], constraints['error_vel_max'])
    
    # Limit attitude error (quaternion vector part)
    e_limited[6:9] = np.clip(e[6:9], -constraints['error_att_max'], constraints['error_att_max'])
    
    # Limit angular velocity error
    e_limited[9:12] = np.clip(e[9:12], -constraints['error_angvel_max'], constraints['error_angvel_max'])
    
    return e_limited


# ============================================================================
# Simulation Functions
# ============================================================================

def simulate_closed_loop(lqr: LQRController, x_ref: np.ndarray, phy_params: PhyParams,
                         constraints: Dict, x0: np.ndarray = None, 
                         t_span: Tuple[float, float] = (0.0, 5.0), 
                         n_points: int = 500,
                         simulation_mode: SimulationMode = SimulationMode.ALL) -> Tuple:
    """
    Simulate closed-loop system with LQR controller, control input constraints, and actuator dynamics.
    
    Args:
        lqr: LQR controller
        x_ref: Reference state
        phy_params: Physical parameters
        constraints: Control input constraints dictionary
        x0: Initial state (default: zeros)
        t_span: Time span for simulation (t_start, t_end)
        n_points: Number of evaluation points
        simulation_mode: Simulation mode to use
    
    Returns:
        sol: Solution from solve_ivp
        x_traj: State trajectory (N, 12)
        e_traj: Error trajectory (N, 12)
        u_traj: Raw control inputs (N, 4)
        u_traj_limited: Limited control inputs (N, 4)
        u_actuator: Actuator outputs after first-order lag (N, 4)
    """
    if x0 is None:
        x0 = np.zeros(12)
    
    A = lqr.A
    B = lqr.B
    K = lqr.get_K()
    
    # Determine which features to enable based on simulation mode
    use_error_limitation = simulation_mode in [
        SimulationMode.POSITION_ERROR_LIMITATION,
        SimulationMode.POSITION_ERROR_LIMITATION_AND_DYNAMICS,
        SimulationMode.ALL
    ]
    
    use_actuator_limitation = simulation_mode in [
        SimulationMode.ACTUATOR_LIMITATION,
        SimulationMode.ACTUATOR_LIMITATION_AND_DYNAMICS,
        SimulationMode.ALL
    ]
    
    use_actuator_dynamics = simulation_mode in [
        SimulationMode.ACTUATOR_DYNAMICS,
        SimulationMode.ACTUATOR_LIMITATION_AND_DYNAMICS,
        SimulationMode.POSITION_ERROR_LIMITATION_AND_DYNAMICS,
        SimulationMode.ALL
    ]
    
    # Extract actuator time constants (use default if not using dynamics)
    if use_actuator_dynamics:
        tau_thrust_angle = constraints['tau_thrust_angle']
        tau_thrust = constraints['tau_thrust']
        tau_yaw_torque = constraints['tau_yaw_torque']
    else:
        # Set very small time constants to effectively disable dynamics
        tau_thrust_angle = 0.001
        tau_thrust = 0.001
        tau_yaw_torque = 0.001
    
    # Extended state: [x(12), u_actuator(4)] if using actuator dynamics, otherwise just [x(12)]
    if use_actuator_dynamics:
        x_extended_0 = np.concatenate([x0, np.zeros(4)])  # Initial actuator outputs are zero
    else:
        x_extended_0 = x0
    
    def system_dynamics(t, x_state):
        """
        System dynamics with optional actuator first-order lag.
        """
        if use_actuator_dynamics:
            # Split extended state
            x = x_state[0:12]  # System state
            u_actuator = x_state[12:16]  # Actuator outputs
        else:
            x = x_state  # System state only
            u_actuator = None
        
        # Compute error
        e = x - x_ref
        
        # Limit state error before computing LQR control law (if enabled)
        if use_error_limitation:
            e_limited = limit_state_error(e, constraints)
        else:
            e_limited = e
        
        # Compute LQR control law using (possibly limited) error
        u_raw = -K @ e_limited
        
        # Apply control input constraints (if enabled)
        if use_actuator_limitation:
            u_cmd = apply_control_constraints(u_raw, phy_params, constraints)
        else:
            u_cmd = u_raw
        
        if use_actuator_dynamics:
            # Actuator dynamics: first-order lag
            # du_act/dt = (u_cmd - u_act) / tau
            u_actuator_dot = np.zeros(4)
            u_actuator_dot[0] = (u_cmd[0] - u_actuator[0]) / tau_thrust_angle  # qx_cmd
            u_actuator_dot[1] = (u_cmd[1] - u_actuator[1]) / tau_thrust_angle  # qy_cmd
            u_actuator_dot[2] = (u_cmd[2] - u_actuator[2]) / tau_thrust  # thrust_cmd
            u_actuator_dot[3] = (u_cmd[3] - u_actuator[3]) / tau_yaw_torque  # r_cmd
            
            # System dynamics: x_dot = A @ x + B @ u_actuator
            # Use actuator outputs (not commands) for system dynamics
            x_dot = A @ x + B @ u_actuator
            
            # Combine derivatives
            x_state_dot = np.concatenate([x_dot, u_actuator_dot])
        else:
            # No actuator dynamics: use command directly
            x_dot = A @ x + B @ u_cmd
            x_state_dot = x_dot
        
        return x_state_dot
    
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(system_dynamics, t_span, x_extended_0, t_eval=t_eval)
    
    # Recover state trajectory and actuator outputs
    if use_actuator_dynamics:
        x_extended_traj = sol.y.T  # shape: (N, 16)
        x_traj = x_extended_traj[:, 0:12]  # System state trajectory
        u_actuator_traj = x_extended_traj[:, 12:16]  # Actuator outputs trajectory
    else:
        x_traj = sol.y.T  # shape: (N, 12)
        u_actuator_traj = None
    
    e_traj = x_traj - x_ref  # Error trajectory
    
    # Calculate control inputs (raw, limited, and actuator outputs) at each time step
    u_traj = np.zeros((len(sol.t), 4))
    u_traj_limited = np.zeros((len(sol.t), 4))
    for i in range(len(sol.t)):
        e = e_traj[i]
        # Limit state error before computing LQR control law (if enabled)
        if use_error_limitation:
            e_limited = limit_state_error(e, constraints)
        else:
            e_limited = e
        u_raw = -K @ e_limited
        u_traj[i] = u_raw
        if use_actuator_limitation:
            u_traj_limited[i] = apply_control_constraints(u_raw, phy_params, constraints)
        else:
            u_traj_limited[i] = u_raw
    
    # If not using actuator dynamics, set u_actuator to u_traj_limited
    if not use_actuator_dynamics:
        u_actuator_traj = u_traj_limited.copy()
    
    return sol, x_traj, e_traj, u_traj, u_traj_limited, u_actuator_traj


def apply_control_constraints(u_raw: np.ndarray, phy_params: PhyParams, 
                              constraints: Dict) -> np.ndarray:
    """
    Apply control input constraints to raw control command.
    
    Args:
        u_raw: Raw control input [qx_cmd, qy_cmd, thrust_cmd, r_cmd]
        phy_params: Physical parameters
        constraints: Control input constraints dictionary
    
    Returns:
        u_limited: Limited control input
    """
    u_limited = np.zeros(4)
    
    # 1. Thrust deflection angle limits
    u_limited[0] = np.clip(u_raw[0], constraints['servo_0_min'], constraints['servo_0_max'])
    u_limited[1] = np.clip(u_raw[1], constraints['servo_1_min'], constraints['servo_1_max'])
    
    # 2. Total thrust limits (thrust-to-weight ratio limitation)
    # Calculate total thrust = thrust_cmd + MASS * G
    thrust_total = u_raw[2] + phy_params.MASS * phy_params.G
    
    # Limit total thrust within [0, 1.5 * MASS * G] range
    thrust_total_limited = np.clip(
        thrust_total,
        constraints['thrust_total_min'],
        constraints['thrust_total_max']
    )
    
    # Convert back to thrust_cmd = total thrust - MASS * G
    u_limited[2] = thrust_total_limited - phy_params.MASS * phy_params.G
    
    # 3. Yaw torque limits
    u_limited[3] = np.clip(u_raw[3], constraints['r_cmd_min'], constraints['r_cmd_max'])
    
    return u_limited


# ============================================================================
# Quaternion to Euler Angle Conversion
# ============================================================================

def quaternion_vector_to_euler(qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    """
    Convert quaternion vector part [qx, qy, qz] to Euler angles (roll, pitch, yaw).
    
    Note: LQR state only contains [qx, qy, qz], so we need to compute qw.
    For small angles: qw ≈ 1, but we compute it properly: qw = sqrt(1 - qx^2 - qy^2 - qz^2)
    
    Args:
        qx: Quaternion x component
        qy: Quaternion y component
        qz: Quaternion z component
    
    Returns:
        (roll, pitch, yaw): Euler angles in radians
    """
    # Compute qw from normalization constraint: qx^2 + qy^2 + qz^2 + qw^2 = 1
    qw_squared = 1.0 - (qx**2 + qy**2 + qz**2)
    if qw_squared < 0:
        # Handle numerical errors, ensure qw_squared >= 0
        qw_squared = 0.0
    qw = np.sqrt(qw_squared)
    
    # Ensure quaternion is normalized (choose positive qw)
    if qw < 0:
        qw = -qw
    
    # Create quaternion [qx, qy, qz, qw] for scipy
    quat = np.array([qx, qy, qz, qw])
    
    # Normalize quaternion
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 0:
        quat = quat / quat_norm
    
    # Convert to rotation object and then to Euler angles
    # Using 'xyz' convention (intrinsic rotations): roll, pitch, yaw
    r = Rotation.from_quat(quat)
    euler = r.as_euler('xyz', degrees=False)  # Returns [roll, pitch, yaw] in radians
    
    return euler[0], euler[1], euler[2]


def quaternion_trajectory_to_euler(q_traj: np.ndarray) -> np.ndarray:
    """
    Convert quaternion trajectory [qx, qy, qz] to Euler angles [roll, pitch, yaw].
    
    Args:
        q_traj: Quaternion trajectory, shape (N, 3) where columns are [qx, qy, qz]
    
    Returns:
        euler_traj: Euler angles trajectory, shape (N, 3) where columns are [roll, pitch, yaw] in radians
    """
    N = q_traj.shape[0]
    euler_traj = np.zeros((N, 3))
    
    for i in range(N):
        roll, pitch, yaw = quaternion_vector_to_euler(q_traj[i, 0], q_traj[i, 1], q_traj[i, 2])
        euler_traj[i, 0] = roll
        euler_traj[i, 1] = pitch
        euler_traj[i, 2] = yaw
    
    return euler_traj


# ============================================================================
# Comparison Simulation Functions
# ============================================================================

def run_comparison_simulations(lqr: LQRController, x_ref: np.ndarray, phy_params: PhyParams,
                               constraints: Dict, x0: np.ndarray = None,
                               t_span: Tuple[float, float] = (0.0, 5.0),
                               n_points: int = 500,
                               modes: List[SimulationMode] = None) -> Dict:
    """
    Run comparison simulations for multiple simulation modes.
    
    Args:
        lqr: LQR controller
        x_ref: Reference state
        phy_params: Physical parameters
        constraints: Control input constraints dictionary
        x0: Initial state (default: zeros)
        t_span: Time span for simulation
        n_points: Number of evaluation points
        modes: List of simulation modes to run (default: all modes)
    
    Returns:
        results: Dictionary mapping mode names to simulation results
    """
    if modes is None:
        modes = list(SimulationMode)
    
    if x0 is None:
        x0 = np.zeros(12)
    
    results = {}
    
    print(f"\nRunning {len(modes)} simulation modes for comparison...")
    for i, mode in enumerate(modes, 1):
        print(f"\n[{i}/{len(modes)}] Running mode: {mode.value}")
        try:
            sol, x_traj, e_traj, u_traj, u_traj_limited, u_actuator = simulate_closed_loop(
                lqr, x_ref, phy_params, constraints, x0, t_span, n_points, mode
            )
            results[mode.value] = {
                'sol': sol,
                'x_traj': x_traj,
                'e_traj': e_traj,
                'u_traj': u_traj,
                'u_traj_limited': u_traj_limited,
                'u_actuator': u_actuator,
                'mode': mode
            }
            print(f"  Completed: {mode.value}")
        except Exception as e:
            print(f"  Error: {mode.value} - {e}")
            continue
    
    return results


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_simulation_results(sol, x_traj: np.ndarray, x_ref: np.ndarray,
                           u_traj: np.ndarray, u_traj_limited: np.ndarray,
                           phy_params: PhyParams, constraints: Dict = None,
                           u_actuator: np.ndarray = None, show_plot: bool = True):
    """
    Plot simulation results: states and control inputs.
    
    Args:
        sol: Solution from solve_ivp
        x_traj: State trajectory (N, 12)
        x_ref: Reference state (12,)
        u_traj: Raw control inputs (N, 4)
        u_traj_limited: Limited control inputs (N, 4)
        phy_params: Physical parameters (for computing total thrust)
        constraints: Control constraints dictionary (for plotting limits)
        u_actuator: Actuator outputs after first-order lag (N, 4), optional
        show_plot: Whether to show the plot (default: True)
    """
    # Create 3x2 subplot layout (3 rows, 2 columns)
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex='col')
    axs = axs.flatten()  # Flatten to 1D array for easier indexing
    
    # Left column: Position, Attitude, Thrust deflection angles
    # Right column: Velocity, Angular velocity, Total thrust and yaw torque
    
    # Left column - Row 0: Position states (x, y, z)
    axs[0].plot(sol.t, x_traj[:, 0], label='x', linewidth=2)
    axs[0].plot(sol.t, x_traj[:, 1], label='y', linewidth=2)
    axs[0].plot(sol.t, x_traj[:, 2], label='z', linewidth=2)
    axs[0].axhline(x_ref[0], linestyle='--', color='r', alpha=0.5, label='x_ref')
    axs[0].axhline(x_ref[1], linestyle='--', color='g', alpha=0.5, label='y_ref')
    axs[0].axhline(x_ref[2], linestyle='--', color='b', alpha=0.5, label='z_ref')
    axs[0].set_ylabel('Position [m]')
    axs[0].legend(loc='upper right', ncol=2)
    axs[0].grid(True)
    
    # Right column - Row 0: Velocity states (vx, vy, vz)
    axs[1].plot(sol.t, x_traj[:, 3], label='vx', linewidth=2)
    axs[1].plot(sol.t, x_traj[:, 4], label='vy', linewidth=2)
    axs[1].plot(sol.t, x_traj[:, 5], label='vz', linewidth=2)
    axs[1].axhline(x_ref[3], linestyle='--', color='r', alpha=0.5, label='vx_ref')
    axs[1].axhline(x_ref[4], linestyle='--', color='g', alpha=0.5, label='vy_ref')
    axs[1].axhline(x_ref[5], linestyle='--', color='b', alpha=0.5, label='vz_ref')
    axs[1].set_ylabel('Velocity [m/s]')
    axs[1].legend(loc='upper right', ncol=2)
    axs[1].grid(True)
    
    # Left column - Row 1: Attitude states (convert from quaternion to Euler angles)
    # Convert trajectory quaternion [qx, qy, qz] to Euler angles [roll, pitch, yaw]
    q_traj = x_traj[:, 6:9]  # Extract [qx, qy, qz]
    euler_traj = quaternion_trajectory_to_euler(q_traj)
    
    # Convert reference quaternion to Euler angles
    q_ref = x_ref[6:9]
    roll_ref, pitch_ref, yaw_ref = quaternion_vector_to_euler(q_ref[0], q_ref[1], q_ref[2])
    
    # Plot Euler angles (in degrees for better readability)
    axs[2].plot(sol.t, np.degrees(euler_traj[:, 0]), label='Roll', linewidth=2)
    axs[2].plot(sol.t, np.degrees(euler_traj[:, 1]), label='Pitch', linewidth=2)
    axs[2].plot(sol.t, np.degrees(euler_traj[:, 2]), label='Yaw', linewidth=2)
    axs[2].axhline(np.degrees(roll_ref), linestyle='--', color='r', alpha=0.5, label='Roll_ref')
    axs[2].axhline(np.degrees(pitch_ref), linestyle='--', color='g', alpha=0.5, label='Pitch_ref')
    axs[2].axhline(np.degrees(yaw_ref), linestyle='--', color='b', alpha=0.5, label='Yaw_ref')
    axs[2].set_ylabel('Attitude [deg]')
    axs[2].legend(loc='upper right', ncol=2)
    axs[2].grid(True)
    
    # Right column - Row 1: Angular velocity states (p, q, r) - convert to degrees per second
    axs[3].plot(sol.t, np.degrees(x_traj[:, 9]), label='p', linewidth=2)
    axs[3].plot(sol.t, np.degrees(x_traj[:, 10]), label='q', linewidth=2)
    axs[3].plot(sol.t, np.degrees(x_traj[:, 11]), label='r', linewidth=2)
    axs[3].axhline(np.degrees(x_ref[9]), linestyle='--', color='r', alpha=0.5, label='p_ref')
    axs[3].axhline(np.degrees(x_ref[10]), linestyle='--', color='g', alpha=0.5, label='q_ref')
    axs[3].axhline(np.degrees(x_ref[11]), linestyle='--', color='b', alpha=0.5, label='r_ref')
    axs[3].set_ylabel('Angular Velocity [deg/s]')
    axs[3].legend(loc='upper right', ncol=2)
    axs[3].grid(True)
    
    # Control inputs - Split into two subplots
    # Calculate total thrust (thrust_cmd + gravity compensation)
    thrust_total_raw = u_traj[:, 2] + phy_params.MASS * phy_params.G
    thrust_total_limited = u_traj_limited[:, 2] + phy_params.MASS * phy_params.G
    
    # Left column - Row 2: Thrust deflection angles (qx_cmd, qy_cmd) - convert to degrees
    # Plot limited commands (before actuator dynamics) with dashed lines
    axs[4].plot(sol.t, np.degrees(u_traj_limited[:, 0]), 'b--', linewidth=2, label='qx_cmd (before actuator)')
    axs[4].plot(sol.t, np.degrees(u_traj_limited[:, 1]), 'r--', linewidth=2, label='qy_cmd (before actuator)')
    
    # Plot actuator outputs (after first-order lag) with solid lines
    if u_actuator is not None:
        axs[4].plot(sol.t, np.degrees(u_actuator[:, 0]), 'b-', linewidth=2, label='qx_actuator (after lag)')
        axs[4].plot(sol.t, np.degrees(u_actuator[:, 1]), 'r-', linewidth=2, label='qy_actuator (after lag)')
    
    axs[4].set_ylabel('Thrust Deflection Angle [deg]')
    axs[4].set_xlabel('Time [s]')
    axs[4].legend(loc='upper right', fontsize=8, ncol=2)
    axs[4].grid(True)
    
    # Right column - Row 2: Total thrust and yaw torque (thrust_total, r_cmd)
    # Plot limited commands (before actuator dynamics) with dashed lines
    axs[5].plot(sol.t, thrust_total_limited, 'g--', linewidth=2, label='thrust_total (before actuator)')
    axs[5].plot(sol.t, u_traj_limited[:, 3], 'm--', linewidth=2, label='r_cmd (before actuator)')
    
    # Plot actuator outputs (after first-order lag) with solid lines
    if u_actuator is not None:
        thrust_total_actuator = u_actuator[:, 2] + phy_params.MASS * phy_params.G
        axs[5].plot(sol.t, thrust_total_actuator, 'g-', linewidth=2, label='thrust_actuator (after lag)')
        axs[5].plot(sol.t, u_actuator[:, 3], 'm-', linewidth=2, label='r_actuator (after lag)')
    
    # Add reference lines for gravity compensation and thrust limits
    mg = phy_params.MASS * phy_params.G
    axs[5].axhline(mg, color='g', linestyle=':', 
                   alpha=0.3, label=f'mg = {mg:.2f} N')
    
    # Add thrust limit lines if constraints are provided
    if constraints is not None:
        axs[5].axhline(constraints['thrust_total_max'], color='r', linestyle=':', 
                      alpha=0.3, label=f'Max thrust = {constraints["thrust_total_max"]:.2f} N')
        axs[5].axhline(constraints['thrust_total_min'], color='b', linestyle=':', 
                      alpha=0.3, label=f'Min thrust = {constraints["thrust_total_min"]:.2f} N')
    
    axs[5].set_ylabel('Total Thrust [N] / Yaw Torque [Nm]')
    axs[5].set_xlabel('Time [s]')
    axs[5].legend(loc='upper right', fontsize=8, ncol=2)
    axs[5].grid(True)
    
    plt.tight_layout()
    if show_plot:
        plt.show()
    
    return fig


def plot_comparison_results(results: Dict, x_ref: np.ndarray, phy_params: PhyParams,
                            constraints: Dict = None, show_plot: bool = True):
    """
    Plot comparison results for multiple simulation modes.
    
    Args:
        results: Dictionary mapping mode names to simulation results
        x_ref: Reference state
        phy_params: Physical parameters
        constraints: Control constraints dictionary (optional)
        show_plot: Whether to show the plot (default: True)
    """
    if not results:
        print("Warning: No results to plot")
        return None
    
    # Define colors and line styles
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Create 3x2 subplot layout
    fig, axs = plt.subplots(3, 2, figsize=(18, 14), sharex='col')
    axs = axs.flatten()
    
    # Get results for all modes
    mode_names = list(results.keys())
    num_modes = len(mode_names)
    
    # Left column - Row 0: Position states (x, y, z)
    for idx, mode_name in enumerate(mode_names):
        result = results[mode_name]
        sol = result['sol']
        x_traj = result['x_traj']
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        label = mode_name
        
        axs[0].plot(sol.t, x_traj[:, 0], color=color, linestyle=linestyle, 
                   linewidth=2, label=f'{label} - x', alpha=0.8)
        axs[0].plot(sol.t, x_traj[:, 1], color=color, linestyle=linestyle, 
                   linewidth=1.5, label=f'{label} - y', alpha=0.6)
        axs[0].plot(sol.t, x_traj[:, 2], color=color, linestyle=linestyle, 
                   linewidth=1.5, label=f'{label} - z', alpha=0.6)
    
    axs[0].axhline(x_ref[0], linestyle='--', color='r', alpha=0.5, linewidth=1)
    axs[0].axhline(x_ref[1], linestyle='--', color='g', alpha=0.5, linewidth=1)
    axs[0].axhline(x_ref[2], linestyle='--', color='b', alpha=0.5, linewidth=1)
    axs[0].set_ylabel('Position [m]')
    axs[0].legend(loc='upper right', fontsize=7, ncol=2)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title('Position Comparison')
    
    # Right column - Row 0: Velocity states (vx, vy, vz)
    for idx, mode_name in enumerate(mode_names):
        result = results[mode_name]
        sol = result['sol']
        x_traj = result['x_traj']
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        label = mode_name
        
        axs[1].plot(sol.t, x_traj[:, 3], color=color, linestyle=linestyle, 
                   linewidth=2, label=f'{label} - vx', alpha=0.8)
        axs[1].plot(sol.t, x_traj[:, 4], color=color, linestyle=linestyle, 
                   linewidth=1.5, label=f'{label} - vy', alpha=0.6)
        axs[1].plot(sol.t, x_traj[:, 5], color=color, linestyle=linestyle, 
                   linewidth=1.5, label=f'{label} - vz', alpha=0.6)
    
    axs[1].axhline(x_ref[3], linestyle='--', color='r', alpha=0.5, linewidth=1)
    axs[1].axhline(x_ref[4], linestyle='--', color='g', alpha=0.5, linewidth=1)
    axs[1].axhline(x_ref[5], linestyle='--', color='b', alpha=0.5, linewidth=1)
    axs[1].set_ylabel('Velocity [m/s]')
    axs[1].legend(loc='upper right', fontsize=7, ncol=2)
    axs[1].grid(True, alpha=0.3)
    axs[1].set_title('Velocity Comparison')
    
    # Left column - Row 1: Attitude states (converted to Euler angles)
    for idx, mode_name in enumerate(mode_names):
        result = results[mode_name]
        sol = result['sol']
        x_traj = result['x_traj']
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        label = mode_name
        
        q_traj = x_traj[:, 6:9]
        euler_traj = quaternion_trajectory_to_euler(q_traj)
        
        axs[2].plot(sol.t, np.degrees(euler_traj[:, 0]), color=color, linestyle=linestyle,
                   linewidth=2, label=f'{label} - Roll', alpha=0.8)
        axs[2].plot(sol.t, np.degrees(euler_traj[:, 1]), color=color, linestyle=linestyle,
                   linewidth=1.5, label=f'{label} - Pitch', alpha=0.6)
        axs[2].plot(sol.t, np.degrees(euler_traj[:, 2]), color=color, linestyle=linestyle,
                   linewidth=1.5, label=f'{label} - Yaw', alpha=0.6)
    
    q_ref = x_ref[6:9]
    roll_ref, pitch_ref, yaw_ref = quaternion_vector_to_euler(q_ref[0], q_ref[1], q_ref[2])
    axs[2].axhline(np.degrees(roll_ref), linestyle='--', color='r', alpha=0.5, linewidth=1)
    axs[2].axhline(np.degrees(pitch_ref), linestyle='--', color='g', alpha=0.5, linewidth=1)
    axs[2].axhline(np.degrees(yaw_ref), linestyle='--', color='b', alpha=0.5, linewidth=1)
    axs[2].set_ylabel('Attitude [deg]')
    axs[2].legend(loc='upper right', fontsize=7, ncol=2)
    axs[2].grid(True, alpha=0.3)
    axs[2].set_title('Attitude Comparison')
    
    # Right column - Row 1: Angular velocity states
    for idx, mode_name in enumerate(mode_names):
        result = results[mode_name]
        sol = result['sol']
        x_traj = result['x_traj']
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        label = mode_name
        
        axs[3].plot(sol.t, np.degrees(x_traj[:, 9]), color=color, linestyle=linestyle,
                   linewidth=2, label=f'{label} - p', alpha=0.8)
        axs[3].plot(sol.t, np.degrees(x_traj[:, 10]), color=color, linestyle=linestyle,
                   linewidth=1.5, label=f'{label} - q', alpha=0.6)
        axs[3].plot(sol.t, np.degrees(x_traj[:, 11]), color=color, linestyle=linestyle,
                   linewidth=1.5, label=f'{label} - r', alpha=0.6)
    
    axs[3].axhline(np.degrees(x_ref[9]), linestyle='--', color='r', alpha=0.5, linewidth=1)
    axs[3].axhline(np.degrees(x_ref[10]), linestyle='--', color='g', alpha=0.5, linewidth=1)
    axs[3].axhline(np.degrees(x_ref[11]), linestyle='--', color='b', alpha=0.5, linewidth=1)
    axs[3].set_ylabel('Angular Velocity [deg/s]')
    axs[3].legend(loc='upper right', fontsize=7, ncol=2)
    axs[3].grid(True, alpha=0.3)
    axs[3].set_title('Angular Velocity Comparison')
    
    # Left column - Row 2: Thrust deflection angles
    for idx, mode_name in enumerate(mode_names):
        result = results[mode_name]
        sol = result['sol']
        u_actuator = result['u_actuator']
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        label = mode_name
        
        if u_actuator is not None:
            axs[4].plot(sol.t, np.degrees(u_actuator[:, 0]), color=color, linestyle=linestyle,
                       linewidth=2, label=f'{label} - qx', alpha=0.8)
            axs[4].plot(sol.t, np.degrees(u_actuator[:, 1]), color=color, linestyle=linestyle,
                       linewidth=1.5, label=f'{label} - qy', alpha=0.6)
    
    axs[4].set_ylabel('Thrust Deflection Angle [deg]')
    axs[4].set_xlabel('Time [s]')
    axs[4].legend(loc='upper right', fontsize=7, ncol=2)
    axs[4].grid(True, alpha=0.3)
    axs[4].set_title('Control Input: Thrust Deflection Angles')
    
    # Right column - Row 2: Total thrust and yaw torque
    for idx, mode_name in enumerate(mode_names):
        result = results[mode_name]
        sol = result['sol']
        u_actuator = result['u_actuator']
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        label = mode_name
        
        if u_actuator is not None:
            thrust_total = u_actuator[:, 2] + phy_params.MASS * phy_params.G
            axs[5].plot(sol.t, thrust_total, color=color, linestyle=linestyle,
                       linewidth=2, label=f'{label} - Thrust', alpha=0.8)
            axs[5].plot(sol.t, u_actuator[:, 3], color=color, linestyle=linestyle,
                       linewidth=1.5, label=f'{label} - r_cmd', alpha=0.6)
    
    mg = phy_params.MASS * phy_params.G
    axs[5].axhline(mg, color='g', linestyle=':', alpha=0.3, linewidth=1, label=f'mg = {mg:.2f} N')
    
    if constraints is not None:
        axs[5].axhline(constraints['thrust_total_max'], color='r', linestyle=':', 
                      alpha=0.3, linewidth=1, label=f'Max = {constraints["thrust_total_max"]:.2f} N')
        axs[5].axhline(constraints['thrust_total_min'], color='b', linestyle=':', 
                      alpha=0.3, linewidth=1, label=f'Min = {constraints["thrust_total_min"]:.2f} N')
    
    axs[5].set_ylabel('Total Thrust [N] / Yaw Torque [Nm]')
    axs[5].set_xlabel('Time [s]')
    axs[5].legend(loc='upper right', fontsize=7, ncol=2)
    axs[5].grid(True, alpha=0.3)
    axs[5].set_title('Control Input: Thrust and Yaw Torque')
    
    plt.suptitle('Simulation Mode Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if show_plot:
        plt.show()
    
    return fig


# ============================================================================
# Mode Selection Helper Functions
# ============================================================================

def get_simulation_mode_interactive() -> Tuple[Optional[SimulationMode], Optional[List[SimulationMode]]]:
    """
    Interactively get user-selected simulation mode(s).
    
    Returns:
        (selected_mode, modes_to_compare): 
            - If single mode: (selected_mode, None)
            - If comparison: (None, list of modes to compare)
    """
    print("\n" + "=" * 70)
    print("Simulation Mode Selection")
    print("=" * 70)
    print("Available simulation modes:")
    mode_list = list(SimulationMode)
    for i, mode in enumerate(mode_list, 1):
        print(f"  {i}. {mode.value}")
    print(f"  {len(mode_list) + 1}. compare - Select modes to compare")
    print(f"  {len(mode_list) + 2}. compare_all - Compare all modes")
    print("=" * 70)
    
    try:
        choice = input(f"\nPlease select option (1-{len(mode_list) + 2}), or press Enter for single mode (1): ").strip()
        
        if choice == "":
            # Default: single mode (first mode)
            return mode_list[0], None
        
        choice_num = int(choice)
        
        if choice_num == len(mode_list) + 2:
            # Compare all modes
            return None, mode_list
        elif choice_num == len(mode_list) + 1:
            # Select modes to compare
            return get_modes_for_comparison_interactive(mode_list)
        elif 1 <= choice_num <= len(mode_list):
            # Select single mode
            selected_mode = mode_list[choice_num - 1]
            return selected_mode, None
        else:
            print(f"Invalid selection, using default: single mode (1)")
            return mode_list[0], None
    except (ValueError, KeyboardInterrupt):
        print(f"\nUsing default: single mode (1)")
        return mode_list[0], None


def get_modes_for_comparison_interactive(mode_list: List[SimulationMode]) -> Tuple[None, List[SimulationMode]]:
    """
    Interactively get multiple modes for comparison.
    
    Args:
        mode_list: List of all available simulation modes
    
    Returns:
        (None, selected_modes): Tuple with None and list of selected modes
    """
    print("\n" + "=" * 70)
    print("Select Modes for Comparison")
    print("=" * 70)
    print("Available simulation modes:")
    for i, mode in enumerate(mode_list, 1):
        print(f"  {i}. {mode.value}")
    print("=" * 70)
    print("Enter mode numbers separated by commas (e.g., 1,3,5)")
    print("Or enter 'all' to compare all modes")
    
    try:
        choice = input("\nPlease select modes: ").strip()
        
        if choice.lower() == "all":
            return None, mode_list
        
        # Parse comma-separated numbers
        selected_indices = [int(x.strip()) for x in choice.split(',')]
        selected_modes = []
        
        for idx in selected_indices:
            if 1 <= idx <= len(mode_list):
                selected_modes.append(mode_list[idx - 1])
            else:
                print(f"Warning: Invalid mode number {idx}, skipping")
        
        if len(selected_modes) < 2:
            print("Warning: At least 2 modes are required for comparison. Using all modes.")
            return None, mode_list
        
        # Remove duplicates while preserving order
        seen = set()
        unique_modes = []
        for mode in selected_modes:
            if mode not in seen:
                seen.add(mode)
                unique_modes.append(mode)
        
        print(f"\nSelected {len(unique_modes)} modes for comparison:")
        for mode in unique_modes:
            print(f"  - {mode.value}")
        
        return None, unique_modes
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nError parsing input: {e}")
        print("Using all modes for comparison")
        return None, mode_list


def get_simulation_mode_from_config(config_mode: Optional[str] = None, 
                                    compare_modes: Optional[List[str]] = None) -> Tuple[Optional[SimulationMode], Optional[List[SimulationMode]]]:
    """
    Get simulation mode from configuration (for non-interactive runs).
    
    Args:
        config_mode: Configuration mode string for single mode, or None
        compare_modes: List of mode names to compare, or None
    
    Returns:
        (selected_mode, modes_to_compare): 
            - If single mode: (selected_mode, None)
            - If comparison: (None, list of modes to compare)
    """
    # If compare_modes is provided, use it
    if compare_modes is not None and len(compare_modes) > 0:
        mode_map = {mode.value: mode for mode in SimulationMode}
        selected_modes = []
        for mode_name in compare_modes:
            if mode_name in mode_map:
                selected_modes.append(mode_map[mode_name])
            elif mode_name.lower() == "all":
                return None, list(SimulationMode)
            else:
                print(f"Warning: Invalid mode name '{mode_name}', skipping")
        
        if len(selected_modes) < 2:
            print("Warning: At least 2 modes are required for comparison. Using all modes.")
            return None, list(SimulationMode)
        
        return None, selected_modes
    
    # Single mode selection
    if config_mode is None:
        return list(SimulationMode)[0], None  # Default to first mode
    
    if config_mode == "compare_all":
        return None, list(SimulationMode)
    
    try:
        mode_map = {mode.value: mode for mode in SimulationMode}
        if config_mode in mode_map:
            return mode_map[config_mode], None
        else:
            # Try to use enum value directly
            return SimulationMode(config_mode), None
    except (ValueError, KeyError):
        print(f"Warning: Invalid simulation mode '{config_mode}', using default: first mode")
        return list(SimulationMode)[0], None


# ============================================================================
# Main Function
# ============================================================================

def main(interactive: bool = True, config_mode: Optional[str] = None, 
         compare_modes: Optional[List[str]] = None):
    """Main function to run LQR controller test and simulation."""
    print("=" * 70)
    print("LQR Controller for TVC Platform - Test and Simulation")
    print("=" * 70)
    
    # Load parameters
    print("\n[1] Loading parameters...")
    phy_params, Q_matrix, R_matrix, constraints = load_parameters()
    
    # Set control input constraints
    print("\n[1.5] Setting control input constraints...")
    # Thrust deflection angle limits (radians)
    servo_0_min = -0.15  # qx_cmd minimum value
    servo_0_max = 0.15   # qx_cmd maximum value
    servo_1_min = -0.15  # qy_cmd minimum value
    servo_1_max = 0.15   # qy_cmd maximum value

    # Total thrust limits based on thrust-to-weight ratio
    # Thrust-to-weight ratio = 1.5, i.e., maximum thrust is 1.5 times gravity, minimum thrust is 0
    thrust_to_weight_ratio = 1.5  # Thrust-to-weight ratio
    thrust_total_min = 0.0  # Minimum total thrust (N)
    thrust_total_max = thrust_to_weight_ratio * phy_params.MASS * phy_params.G  # Maximum total thrust (N)
    
    # Since thrust_cmd = total thrust - MASS * G, we have:
    # thrust_cmd_min = 0 - MASS * G = -MASS * G
    # thrust_cmd_max = 1.5 * MASS * G - MASS * G = 0.5 * MASS * G
    thrust_cmd_min = thrust_total_min - phy_params.MASS * phy_params.G
    thrust_cmd_max = thrust_total_max - phy_params.MASS * phy_params.G

    # Yaw torque limits (Nm)
    torque_z_min = -0.5
    torque_z_max = 0.5
    r_cmd_min = torque_z_min
    r_cmd_max = torque_z_max

    # Add control input constraints to dictionary
    constraints['servo_0_min'] = servo_0_min
    constraints['servo_0_max'] = servo_0_max
    constraints['servo_1_min'] = servo_1_min
    constraints['servo_1_max'] = servo_1_max
    constraints['thrust_total_min'] = thrust_total_min
    constraints['thrust_total_max'] = thrust_total_max
    constraints['thrust_cmd_min'] = thrust_cmd_min
    constraints['thrust_cmd_max'] = thrust_cmd_max
    constraints['r_cmd_min'] = r_cmd_min
    constraints['r_cmd_max'] = r_cmd_max
    
    print(f"  Thrust deflection angles: [{servo_0_min:.3f}, {servo_0_max:.3f}] rad")
    print(f"  Total thrust: [{thrust_total_min:.2f}, {thrust_total_max:.2f}] N")
    print(f"  Yaw torque: [{r_cmd_min:.3f}, {r_cmd_max:.3f}] Nm")
    
    # Set target state (reference state for LQR controller)
    print("\n[1.6] Setting target state...")
    target_position = np.array([3.0, 0.0, 3.0])  # Target position [x, y, z] in meters
    target_velocity = np.array([0.0, 0.0, 0.0])  # Target velocity [vx, vy, vz] in m/s
    target_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # Target orientation [qx, qy, qz, qw]
    target_angular_velocity = np.array([0.0, 0.0, 0.0])  # Target angular velocity [p, q, r] in rad/s
    
    # Note: LQR state is 12-dimensional, does not include q_w, only uses [qx, qy, qz]
    x_ref = np.concatenate([
        target_position,        # x, y, z
        target_velocity,        # vx, vy, vz
        target_quaternion[:3],  # qx, qy, qz
        target_angular_velocity # p, q, r
    ])
    
    # Set state error limits (for limiting error before LQR computation)
    print("\n[1.7] Setting error limits and actuator parameters...")
    # Limits on position error [m]
    error_pos_max = np.array([1.0, 1.0, 1.0])  # Maximum position error in x, y, z
    # Limits on velocity error [m/s]
    error_vel_max = np.array([5.0, 5.0, 5.0])  # Maximum velocity error in vx, vy, vz
    # Limits on attitude error (quaternion vector part)
    error_att_max = np.array([0.5, 0.5, 0.5])  # Maximum attitude error in qx, qy, qz
    # Limits on angular velocity error [rad/s]
    error_angvel_max = np.array([2.0, 2.0, 2.0])  # Maximum angular velocity error in p, q, r
    
    # Actuator dynamics (first-order lag) time constants [s]
    # Used to simulate real actuator response
    tau_thrust_angle = 0.1  # Time constant for thrust deflection angles (qx_cmd, qy_cmd) [s]
    tau_thrust = 0.5  # Time constant for thrust command [s]
    tau_yaw_torque = 0.05  # Time constant for yaw torque (r_cmd) [s]
    
    # Add error limits and actuator parameters to constraints dictionary
    constraints['error_pos_max'] = error_pos_max
    constraints['error_vel_max'] = error_vel_max
    constraints['error_att_max'] = error_att_max
    constraints['error_angvel_max'] = error_angvel_max
    constraints['tau_thrust_angle'] = tau_thrust_angle
    constraints['tau_thrust'] = tau_thrust
    constraints['tau_yaw_torque'] = tau_yaw_torque
    
    print(f"  Position error limits: {error_pos_max} m")
    print(f"  Velocity error limits: {error_vel_max} m/s")
    print(f"  Attitude error limits: {error_att_max}")
    print(f"  Angular velocity error limits: {error_angvel_max} rad/s")
    print(f"  Actuator time constants:")
    print(f"    Thrust angle: {tau_thrust_angle:.3f} s")
    print(f"    Thrust: {tau_thrust:.3f} s")
    print(f"    Yaw torque: {tau_yaw_torque:.3f} s")
    
    print_parameters(phy_params, x_ref, constraints)
    
    # Setup LQR controller
    print("\n[2] Setting up LQR controller...")
    lqr, success = setup_lqr_controller(phy_params, Q_matrix, R_matrix)
    print_lqr_results(lqr, success)
    
    if not success:
        print("ERROR: Failed to solve LQR. Exiting.")
        return
    
    # Test control law
    print("\n[3] Testing control law...")
    test_control_law(lqr, x_ref)
    
    # Simulation mode selection
    print("\n[4] Simulation Mode Selection")
    
    if interactive:
        selected_mode, modes_to_compare = get_simulation_mode_interactive()
    else:
        selected_mode, modes_to_compare = get_simulation_mode_from_config(config_mode, compare_modes)
    
    if modes_to_compare is not None:
        # Run comparison mode
        print(f"\n[4.1] Running {len(modes_to_compare)} modes for comparison...")
        print("Modes to compare:")
        for mode in modes_to_compare:
            print(f"  - {mode.value}")
        
        results = run_comparison_simulations(
            lqr, x_ref, phy_params, constraints,
            x0=np.zeros(12),
            t_span=(0.0, 10.0),
            n_points=500,
            modes=modes_to_compare
        )
        
        print(f"\n[5] Plotting comparison results...")
        plot_comparison_results(results, x_ref, phy_params, constraints)
        
    else:
        # Run single mode
        if selected_mode is None:
            print("Error: No valid simulation mode selected")
            return
        
        print(f"\n[4.1] Running mode: {selected_mode.value}")
        sol, x_traj, e_traj, u_traj, u_traj_limited, u_actuator = simulate_closed_loop(
            lqr, x_ref, phy_params, constraints,
            x0=np.zeros(12),
            t_span=(0.0, 10.0),
            n_points=500,
            simulation_mode=selected_mode
        )
        print(f"Simulation completed: {len(sol.t)} time points")
        
        print(f"\n[5] Plotting results...")
        plot_simulation_results(sol, x_traj, x_ref, u_traj, u_traj_limited, 
                              phy_params, constraints, u_actuator)
    
    print("\n" + "=" * 70)
    print("All tasks completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    # You can modify here to set non-interactive mode and configuration
    # interactive=True: Interactive mode selection
    # interactive=False, config_mode="mode_name": Use configured single mode
    # interactive=False, compare_modes=["mode1", "mode2", ...]: Compare selected modes
    # config_mode can be: "no_limitation", "actuator_limitation", "position_error_limitation",
    #                    "actuator_dynamics", "actuator_limitation_and_dynamics",
    #                    "position_error_limitation_and_dynamics", "all"
    # compare_modes can be: ["no_limitation", "actuator_limitation"], ["all"], etc.
    
    # Example 1: Interactive mode (default)
    main(interactive=True)
    
    # Example 2: Non-interactive, compare selected modes
    # main(interactive=False, compare_modes=["no_limitation", "actuator_limitation", "all"])
    
    # Example 3: Non-interactive, compare all modes
    # main(interactive=False, compare_modes=["all"])
    
    # Example 4: Non-interactive, run single mode
    # main(interactive=False, config_mode="no_limitation")

