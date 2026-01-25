#!/usr/bin/env python3
"""
Full-State LQR Controller

This module implements a full-state LQR controller that controls all 12 states:
position, velocity, attitude, and angular velocity.
Similar to the lqr_controller_node implementation.
"""

import numpy as np
import scipy.linalg
from typing import Optional
import sys
import os

# Handle both relative and absolute imports
if __name__ == "__main__" or __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from controller_comparison.rocket_dynamics import PhyParams, RocketDynamics
else:
    from .rocket_dynamics import PhyParams, RocketDynamics


class LQRFullStateController:
    """
    Full-state LQR controller for TVC rocket.
    
    State vector (12D): [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
    Control input (4D): [phi, theta, thrust, tau_r]
    
    Coordinate System:
    - Input states: ENU (East-North-Up) - from RocketDynamics
    - Internal LQR computation: NED (North-East-Down) - aligned with original lqr_controller_node
    - Output control: Same for both (phi, theta, thrust, tau_r don't need conversion)
    """
    
    def __init__(self, phy_params: PhyParams, Q: Optional[np.ndarray] = None, 
                 R: Optional[np.ndarray] = None, use_ned: bool = True):
        """
        Initialize full-state LQR controller.
        
        Args:
            phy_params: Physical parameters
            Q: State weighting matrix (12x12), if None uses default
            R: Control weighting matrix (4x4), if None uses default
            use_ned: If True, use NED coordinate system for LQR computation (aligned with original)
                    If False, use ENU coordinate system (current RocketDynamics)
        """
        self.params = phy_params
        self.dynamics = RocketDynamics(phy_params)
        self.use_ned = use_ned
        
        if use_ned:
            # Use NED coordinate system (aligned with original lqr_controller_node)
            # Linearize in NED frame
            self.A, self.B = self._linearize_ned()
        else:
            # Use ENU coordinate system (current RocketDynamics)
            # Linearize around equilibrium
            self.A, self.B = self.dynamics.linearize()
        
        # Set default Q and R if not provided
        if Q is None:
            Q_diag = np.array([1.0, 1.0, 1.0,  # position
                               1.0, 1.0, 1.0,  # velocity
                               1.0, 1.0, 0.1,  # attitude (lower weight on qz)
                               1.0, 1.0, 0.01])  # angular velocity (lower weight on r)
            self.Q = np.diag(Q_diag)
        else:
            self.Q = Q
            
        if R is None:
            R_diag = np.array([10.0, 10.0, 1.0, 10.0])  # [phi, theta, thrust, tau_r]
            self.R = np.diag(R_diag)
        else:
            self.R = R
        
        # Solve LQR
        self.K = None
        self.P = None
        self.solve_lqr()
    
    def solve_lqr(self) -> bool:
        """
        Solve the continuous-time LQR problem.
        
        Solves: A'P + PA - PBR^(-1)B'P + Q = 0
        Gain: K = R^(-1)B'P
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            self.K = np.linalg.solve(self.R, self.B.T @ self.P)
            return True
        except Exception as e:
            print(f"Error solving LQR: {e}")
            self.K = None
            self.P = None
            return False
    
    def _enu_to_ned(self, state_enu: np.ndarray) -> np.ndarray:
        """
        Convert state from ENU (East-North-Up) to NED (North-East-Down).
        
        Args:
            state_enu: State vector in ENU frame [x_enu, y_enu, z_enu, vx_enu, vy_enu, vz_enu, qx, qy, qz, p, q, r]
            
        Returns:
            State vector in NED frame [x_ned, y_ned, z_ned, vx_ned, vy_ned, vz_ned, qx, qy, qz, p, q, r]
        """
        state_ned = state_enu.copy()
        
        # Position: ENU -> NED
        # NED: x_ned = y_enu (North), y_ned = x_enu (East), z_ned = -z_enu (Down = -Up)
        state_ned[0] = state_enu[1]   # x_ned = y_enu (North)
        state_ned[1] = state_enu[0]   # y_ned = x_enu (East)
        state_ned[2] = -state_enu[2]  # z_ned = -z_enu (Down = -Up)
        
        # Velocity: ENU -> NED
        # NED: vx_ned = vy_enu (North velocity), vy_ned = vx_enu (East velocity), vz_ned = -vz_enu (Down = -Up)
        state_ned[3] = state_enu[4]   # vx_ned = vy_enu (North velocity)
        state_ned[4] = state_enu[3]   # vy_ned = vx_enu (East velocity)
        state_ned[5] = -state_enu[5]  # vz_ned = -vz_enu (Down = -Up)
        
        # Attitude and angular velocity: unchanged (qx, qy, qz, p, q, r)
        # state_ned[6:12] = state_enu[6:12]  # Already copied
        
        return state_ned
    
    def _linearize_ned(self) -> tuple:
        """
        Linearize the dynamics in NED coordinate system (aligned with original lqr_controller_node).
        
        Returns:
            A: 12x12 state matrix in NED frame
            B: 12x4 control input matrix in NED frame
        """
        # NED linearization (based on original lqr.py implementation)
        # State vector: [x_ned, y_ned, z_ned, vx_ned, vy_ned, vz_ned, qx, qy, qz, p, q, r]
        # Control input: [phi, theta, thrust, tau_r]
        
        g = self.params.G
        mass = self.params.MASS
        I_XX = self.params.I_XX
        I_YY = self.params.I_YY
        I_ZZ = self.params.I_ZZ
        l = self.params.DIST_COM_2_THRUST
        
        # Linearized A matrix (12x12) in NED frame
        A = np.zeros((12, 12))
        
        # Position dynamics: pos_dot = vel
        A[0:3, 3:6] = np.eye(3)
        
        # Velocity dynamics: vel_dot = f(attitude, thrust)
        # For small angles: qx ≈ roll, qy ≈ pitch
        # In NED: vx_dot (North) ≈ -2g * qy (pitch), vy_dot (East) ≈ 2g * qx (roll)
        A[3, 7] = -2 * g  # vx_dot (North) ≈ -2g * qy (pitch)
        A[4, 6] = 2 * g   # vy_dot (East) ≈ 2g * qx (roll)
        # vz_dot (Down): handled in B matrix (thrust control)
        
        # Quaternion kinematics: q_vec_dot ≈ 0.5 * omega (for small angles)
        A[6, 9] = 0.5   # qx_dot = 0.5 * p
        A[7, 10] = 0.5  # qy_dot = 0.5 * q
        A[8, 11] = 0.5  # qz_dot = 0.5 * r
        
        # Angular velocity dynamics: omega_dot = f(torque)
        # Handled in B matrix
        
        # Linearized B matrix (12x4) in NED frame
        B = np.zeros((12, 4))
        
        # Velocity dynamics: affected by thrust deflection and magnitude
        # In NED: phi affects vx (North), theta affects vy (East)
        B[3, 0] = -g        # vx_dot (North) from phi
        B[4, 1] = g         # vy_dot (East) from theta
        B[5, 2] = 1.0 / mass  # vz_dot (Down) from thrust
        
        # Angular velocity dynamics: affected by deflection and yaw torque
        B[9, 0] = -l * mass * g / I_XX   # p_dot from phi
        B[10, 1] = -l * mass * g / I_YY  # q_dot from theta
        B[11, 3] = 1.0 / I_ZZ             # r_dot from tau_r
        
        return A, B
    
    def compute_control(self, state: np.ndarray, state_ref: np.ndarray) -> np.ndarray:
        """
        Compute control input using LQR feedback law.
        
        Args:
            state: Current state in ENU frame [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
            state_ref: Reference state in ENU frame [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
            
        Returns:
            Control input [phi, theta, thrust, tau_r] (same for both ENU and NED)
        """
        if self.K is None:
            raise ValueError("LQR gain matrix K is not computed. Call solve_lqr() first.")
        
        if self.use_ned:
            # Convert ENU states to NED for LQR computation
            state_ned = self._enu_to_ned(state)
            state_ref_ned = self._enu_to_ned(state_ref)
            
            # Compute state error in NED frame
            error_ned = state_ned - state_ref_ned
            
            # LQR control law in NED: u = u_eq - K * error
            # However, for Z-axis in NED, the direction is opposite to ENU
            # So we need to flip the sign for Z and vz related gains
            # Create a modified error vector with Z and vz signs flipped
            error_ned_modified = error_ned.copy()
            error_ned_modified[2] = -error_ned[2]  # Z position: flip sign
            error_ned_modified[5] = -error_ned[5]  # Z velocity: flip sign
            
            # Apply LQR control law
            u = -self.K @ error_ned_modified
            
            # Add equilibrium control input
            # In NED: equilibrium thrust = mg (to balance gravity)
            # Original implementation: thrust_gimbal_z_frame = u_lqr[2] - MASS * G
            # This means u_lqr[2] = thrust_gimbal_z_frame + mg
            # So we add mg here to match the original behavior
            u_eq = np.array([0.0, 0.0, self.params.MASS * self.params.G, 0.0])
            u = u + u_eq
        else:
            # Use ENU coordinate system (current RocketDynamics)
            # Compute state error
            error = state - state_ref
            
            # LQR control law: u = u_eq - K * error
            u = -self.K @ error
            
            # Add equilibrium control input
            u_eq = np.array([0.0, 0.0, self.params.MASS * self.params.G, 0.0])
            u = u + u_eq
        
        return u
    
    def get_gain_matrix(self) -> Optional[np.ndarray]:
        """Get the LQR gain matrix K (4x12)."""
        return self.K
    
    def get_riccati_solution(self) -> Optional[np.ndarray]:
        """Get the Riccati equation solution P (12x12)."""
        return self.P
