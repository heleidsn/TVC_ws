#!/usr/bin/env python3
"""
Rocket Dynamics and Kinematics Simulation Module

This module implements the 6-DOF dynamics and kinematics for a TVC (Thrust Vector Control) rocket.
The state vector includes position, velocity, attitude (quaternion), and angular velocity.
"""

import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from typing import Tuple, Optional


@dataclass
class PhyParams:
    """Physical parameters for the rocket"""
    MASS: float  # kg
    G: float  # m/s^2, gravitational acceleration
    I_XX: float  # kg*m^2, moment of inertia about X-axis
    I_YY: float  # kg*m^2, moment of inertia about Y-axis
    I_ZZ: float  # kg*m^2, moment of inertia about Z-axis
    DIST_COM_2_THRUST: float  # m, distance from center of mass to thrust point


class RocketDynamics:
    """
    Rocket dynamics and kinematics simulator.
    
    Coordinate System: ENU (East-North-Up) - ROS standard
    - X-axis: East (positive eastward)
    - Y-axis: North (positive northward)
    - Z-axis: Up (positive upward, directly represents altitude)
    - Note: In ENU frame, Z = 0 is ground level, Z > 0 means above ground (higher altitude)
    
    State vector (12D): [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
    - Position: x, y, z (m) in ENU frame
      * z > 0: above ground (e.g., z = 1.0 means 1m altitude)
      * z = 0: at ground level
      * z < 0: below ground (not physically meaningful for flight)
    - Velocity: vx, vy, vz (m/s) in ENU frame
      * vz > 0: upward velocity (climbing)
      * vz < 0: downward velocity (descending)
    - Attitude: qx, qy, qz (quaternion vector part, qw = sqrt(1 - qx^2 - qy^2 - qz^2))
    - Angular velocity: p, q, r (rad/s) in FRD body frame
    
    Control input (4D): [phi, theta, thrust, tau_r]
    - phi: Thrust deflection angle affecting X-acceleration (corresponds to pitch, qy) (rad)
    - theta: Thrust deflection angle affecting Y-acceleration (corresponds to roll, qx) (rad)
    - thrust: Total thrust force (N), positive value
    - tau_r: Yaw torque (Nm)
    - Note: Based on linearize function, phi affects vx (X-accel), theta affects vy (Y-accel)
    """
    
    def __init__(self, phy_params: PhyParams):
        """
        Initialize rocket dynamics.
        
        Args:
            phy_params: Physical parameters of the rocket
        """
        self.params = phy_params
        self.mass = phy_params.MASS
        self.g = phy_params.G
        self.I = np.diag([phy_params.I_XX, phy_params.I_YY, phy_params.I_ZZ])
        self.I_inv = np.linalg.inv(self.I)
        self.l = phy_params.DIST_COM_2_THRUST
        
    def quaternion_to_rotation_matrix(self, q_vec: np.ndarray) -> np.ndarray:
        """
        Convert quaternion vector part to rotation matrix.
        
        Args:
            q_vec: Quaternion vector part [qx, qy, qz]
            
        Returns:
            3x3 rotation matrix from body to world frame
        """
        qw = np.sqrt(1.0 - np.clip(np.sum(q_vec**2), 0.0, 1.0))
        q = np.array([qw, q_vec[0], q_vec[1], q_vec[2]])  # [w, x, y, z]
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x, y, z, w]
        return rot.as_matrix()
    
    def compute_dynamics(self, t: float, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute state derivatives for the rocket dynamics.
        
        Args:
            t: Current time (s)
            state: Current state vector [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
            u: Control input [phi, theta, thrust, tau_r]
            
        Returns:
            State derivative vector
        """
        # Extract state components
        pos = state[0:3]  # [x, y, z]
        vel = state[3:6]  # [vx, vy, vz]
        q_vec = state[6:9]  # [qx, qy, qz]
        omega = state[9:12]  # [p, q, r]
        
        # Extract control inputs
        phi = u[0]  # Deflection angle affecting X-acceleration (corresponds to pitch)
        theta = u[1]  # Deflection angle affecting Y-acceleration (corresponds to roll)
        thrust = u[2]  # Total thrust
        tau_r = u[3]  # Yaw torque
        
        # Compute quaternion scalar part
        qw = np.sqrt(np.clip(1.0 - np.sum(q_vec**2), 0.0, 1.0))
        
        # Rotation matrix from body to world (ENU frame)
        R_bw = self.quaternion_to_rotation_matrix(q_vec)
        
        # Thrust vector in body frame (assuming thrust along body +z axis in ENU)
        # With deflection angles phi and theta
        # In ENU: thrust points upward (positive Z direction)
        # Based on linearize function:
        #   - phi affects vx (X-acceleration): phi>0 -> vx_dot<0, so phi<0 for +X accel
        #   - theta affects vy (Y-acceleration): theta>0 -> vy_dot>0, so theta>0 for +Y accel
        # The body frame axes mapping: body X affects world Y, body Y affects world X
        # So: phi (affects X) -> body Y component, theta (affects Y) -> body X component
        thrust_body = np.array([
            thrust * np.sin(theta),  # body X component from theta (affects Y-accel)
            -thrust * np.sin(phi),   # body Y component from phi (affects X-accel, negative for correct direction)
            thrust * np.cos(phi) * np.cos(theta)  # Positive for upward thrust
        ])
        
        # Convert thrust to world frame
        thrust_world = R_bw @ thrust_body
        
        # Gravity force in world frame (ENU: negative Z direction, downward)
        # In ENU frame, gravity points downward (negative Z direction)
        gravity_world = np.array([0.0, 0.0, -self.g])
        
        # Translational dynamics: F = ma
        # Total acceleration = thrust acceleration + gravity acceleration
        # In ENU frame: gravity is -g (downward), thrust is +thrust/mass (upward)
        # So: accel = thrust/mass + (-g) = thrust/mass - g
        # When thrust=0: accel = 0 - g = -g (downward, correct)
        # When thrust=mg: accel = g - g = 0 (hover, correct)
        # When thrust>mg: accel > 0 (upward, correct)
        accel_world = (thrust_world / self.mass) + gravity_world
        
        # Quaternion kinematics: q_dot = 0.5 * q * [0, omega]
        # For small angles, simplified: q_vec_dot = 0.5 * omega (approximately)
        # More accurate: q_vec_dot = 0.5 * (qw * omega + cross(omega, q_vec))
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        q_vec_dot = 0.5 * np.array([
            qw * omega[0] + omega[1] * q_vec[2] - omega[2] * q_vec[1],
            qw * omega[1] + omega[2] * q_vec[0] - omega[0] * q_vec[2],
            qw * omega[2] + omega[0] * q_vec[1] - omega[1] * q_vec[0]
        ])
        
        # Torque from thrust deflection
        # Thrust creates torque about COM: tau = r x F
        r_thrust = np.array([0.0, 0.0, -self.l])  # Thrust point relative to COM in body frame
        tau_thrust = np.cross(r_thrust, thrust_body)
        
        # Total torque in body frame
        tau_total = np.array([
            tau_thrust[0],  # Torque from theta deflection (affects Y-accel)
            tau_thrust[1],  # Torque from phi deflection (affects X-accel)
            tau_r  # Yaw torque (direct control)
        ])
        
        # Rotational dynamics: tau = I * alpha + omega x (I * omega)
        I_omega = self.I @ omega
        omega_cross_I_omega = np.cross(omega, I_omega)
        alpha = self.I_inv @ (tau_total - omega_cross_I_omega)
        
        # Combine all derivatives
        state_dot = np.concatenate([
            vel,  # pos_dot = vel
            accel_world,  # vel_dot = accel
            q_vec_dot,  # quaternion_dot
            alpha  # omega_dot = alpha
        ])
        
        return state_dot
    
    def linearize(self, state_eq: Optional[np.ndarray] = None, 
                  u_eq: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the dynamics around an equilibrium point.
        
        Args:
            state_eq: Equilibrium state (default: hover at origin)
            u_eq: Equilibrium control input (default: thrust = mg, no deflection)
            
        Returns:
            A: 12x12 state matrix
            B: 12x4 control input matrix
        """
        if state_eq is None:
            # Equilibrium: hover at origin, no rotation
            state_eq = np.zeros(12)
            state_eq[2] = 0.0  # z = 0 (at ground level in ENU)
        
        if u_eq is None:
            # Equilibrium control: thrust balances gravity, no deflection
            u_eq = np.array([0.0, 0.0, self.mass * self.g, 0.0])
        
        # Linearized A matrix (12x12)
        # For small angles, we can approximate the dynamics
        A = np.zeros((12, 12))
        
        # Position dynamics: pos_dot = vel
        A[0:3, 3:6] = np.eye(3)
        
        # Velocity dynamics: vel_dot = f(attitude, thrust)
        # For small angles: qx ≈ roll, qy ≈ pitch
        A[3, 7] = -2 * self.g  # vx_dot ≈ -2g * qy (pitch)
        A[4, 6] = 2 * self.g  # vy_dot ≈ 2g * qx (roll)
        # vz_dot: handled in B matrix (thrust control)
        
        # Quaternion kinematics: q_vec_dot ≈ 0.5 * omega (for small angles)
        A[6, 9] = 0.5  # qx_dot = 0.5 * p
        A[7, 10] = 0.5  # qy_dot = 0.5 * q
        A[8, 11] = 0.5  # qz_dot = 0.5 * r
        
        # Angular velocity dynamics: omega_dot = f(torque)
        # Linearized around equilibrium (no rotation): omega_dot = I^-1 * tau
        # This is handled in B matrix
        
        # Linearized B matrix (12x4)
        B = np.zeros((12, 4))
        
        # Velocity dynamics: affected by thrust deflection and magnitude
        B[3, 0] = -self.g  # vx_dot from phi (affects X-acceleration)
        B[4, 1] = self.g  # vy_dot from theta (affects Y-acceleration)
        B[5, 2] = 1.0 / self.mass  # vz_dot from thrust
        
        # Angular velocity dynamics: affected by deflection and yaw torque
        B[9, 0] = -self.l * self.mass * self.g / self.params.I_XX  # p_dot from phi
        B[10, 1] = -self.l * self.mass * self.g / self.params.I_YY  # q_dot from theta
        B[11, 3] = 1.0 / self.params.I_ZZ  # r_dot from tau_r
        
        return A, B
    
    def simulate_step(self, state: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform one integration step using Runge-Kutta 4th order.
        
        Args:
            state: Current state
            u: Control input
            dt: Time step (s)
            
        Returns:
            Next state
        """
        k1 = self.compute_dynamics(0.0, state, u)
        k2 = self.compute_dynamics(0.0, state + 0.5 * dt * k1, u)
        k3 = self.compute_dynamics(0.0, state + 0.5 * dt * k2, u)
        k4 = self.compute_dynamics(0.0, state + dt * k3, u)
        
        state_next = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize quaternion
        q_vec = state_next[6:9]
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1.0:
            state_next[6:9] = q_vec / q_norm
        
        return state_next
