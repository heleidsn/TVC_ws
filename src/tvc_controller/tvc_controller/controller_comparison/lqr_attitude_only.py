#!/usr/bin/env python3
"""
Attitude-Only LQR Controller

This module implements an LQR controller that only controls the attitude
(orientation and angular velocity) of the rocket. Position and velocity
are not directly controlled by this controller.
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


class LQRAttitudeOnlyController:
    """
    Attitude-only LQR controller for TVC rocket.
    
    This controller only controls:
    - Attitude: qx, qy, qz (6D reduced to 3D for small angles: roll, pitch, yaw)
    - Angular velocity: p, q, r
    
    State vector (6D): [qx, qy, qz, p, q, r] or [roll, pitch, yaw, p, q, r]
    Control input (3D): [phi, theta, tau_r] (thrust is not controlled)
    """
    
    def __init__(self, phy_params: PhyParams, Q_att: Optional[np.ndarray] = None,
                 R_att: Optional[np.ndarray] = None):
        """
        Initialize attitude-only LQR controller.
        
        Args:
            phy_params: Physical parameters
            Q_att: State weighting matrix (6x6) for attitude states, if None uses default
            R_att: Control weighting matrix (3x3) for attitude control, if None uses default
        """
        self.params = phy_params
        self.dynamics = RocketDynamics(phy_params)
        
        # Get full linearized system
        A_full, B_full = self.dynamics.linearize()
        
        # Extract attitude subsystem (rows/cols 6-11 for attitude and angular velocity)
        # State: [qx, qy, qz, p, q, r]
        self.A = A_full[6:12, 6:12]
        
        # Control inputs: [phi, theta, tau_r] (exclude thrust)
        # B matrix columns: [phi, theta, thrust, tau_r]
        # We only use columns 0, 1, 3 (phi, theta, tau_r)
        B_att_full = B_full[6:12, :]  # Attitude rows
        self.B = np.hstack([B_att_full[:, 0:1], B_att_full[:, 1:2], B_att_full[:, 3:4]])
        
        # Set default Q and R if not provided
        if Q_att is None:
            Q_diag = np.array([1.0, 1.0, 0.1,  # attitude (qx, qy, qz)
                               1.0, 1.0, 0.01])  # angular velocity (p, q, r)
            self.Q = np.diag(Q_diag)
        else:
            self.Q = Q_att
            
        if R_att is None:
            R_diag = np.array([10.0, 10.0, 10.0])  # [phi, theta, tau_r]
            self.R = np.diag(R_diag)
        else:
            self.R = R_att
        
        # Solve LQR
        self.K = None
        self.P = None
        self.solve_lqr()
    
    def solve_lqr(self) -> bool:
        """
        Solve the continuous-time LQR problem for attitude subsystem.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            self.K = np.linalg.solve(self.R, self.B.T @ self.P)
            return True
        except Exception as e:
            print(f"Error solving attitude LQR: {e}")
            self.K = None
            self.P = None
            return False
    
    def compute_control(self, state: np.ndarray, state_ref: np.ndarray,
                        thrust_cmd: float = None) -> np.ndarray:
        """
        Compute control input using attitude-only LQR feedback law.
        
        Args:
            state: Full state [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
            state_ref: Full reference state
            thrust_cmd: Thrust command (N). If None, uses equilibrium thrust (mg)
            
        Returns:
            Control input [phi, theta, thrust, tau_r]
        """
        if self.K is None:
            raise ValueError("LQR gain matrix K is not computed. Call solve_lqr() first.")
        
        # Extract attitude states
        att_state = state[6:12]  # [qx, qy, qz, p, q, r]
        att_state_ref = state_ref[6:12]
        
        # Compute attitude error
        att_error = att_state - att_state_ref
        
        # LQR control law for attitude: u_att = -K * att_error
        u_att = -self.K @ att_error  # [phi, theta, tau_r]
        
        # Set thrust command
        if thrust_cmd is None:
            thrust = self.params.MASS * self.params.G  # Equilibrium thrust
        else:
            thrust = thrust_cmd
        
        # Combine into full control vector [phi, theta, thrust, tau_r]
        u = np.array([u_att[0], u_att[1], thrust, u_att[2]])
        
        return u
    
    def get_gain_matrix(self) -> Optional[np.ndarray]:
        """Get the LQR gain matrix K (3x6)."""
        return self.K
    
    def get_riccati_solution(self) -> Optional[np.ndarray]:
        """Get the Riccati equation solution P (6x6)."""
        return self.P
