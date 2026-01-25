#!/usr/bin/env python3
"""
PID Controller for TVC Rocket

This module implements a PID controller for the TVC rocket.
It uses separate PID controllers for position, velocity, attitude, and angular velocity.
"""

import numpy as np
from typing import Optional
import sys
import os

# Handle both relative and absolute imports
if __name__ == "__main__" or __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from controller_comparison.rocket_dynamics import PhyParams
else:
    from .rocket_dynamics import PhyParams


class PIDController:
    """
    PID controller for TVC rocket.
    
    Uses PX4-style P+PID structure:
    - Position loop: P + PID -> velocity command
      * P term: Kp_pos * error_pos (direct feedforward)
      * PID term: PID(error_pos) (feedback)
      * vel_cmd = P_term + PID_term
    - Attitude loop: P + PID -> angular velocity command
      * P term: Kp_att * error_att (direct feedforward)
      * PID term: PID(error_att) (feedback)
      * omega_cmd = P_term + PID_term
    - Velocity PID -> attitude reference (for horizontal control)
    - Angular velocity PID -> control input
    """
    
    def __init__(self, phy_params: PhyParams):
        """
        Initialize PID controller.
        
        Args:
            phy_params: Physical parameters
        """
        self.params = phy_params
        
        # Position control gains [x, y, z]
        # P+PID structure: P term (direct) + PID term (feedback)
        self.Kp_pos = np.array([1.0, 1.0, 10.0])  # P gain for direct feedforward
        self.Kp_pos_pid = np.array([0.8, 0.8, 8.0])  # P gain for PID term
        self.Ki_pos = np.array([0.0, 0.0, 0.0])  # I gain for PID term
        self.Kd_pos = np.array([0.5, 0.5, 0.5])  # D gain for PID term
        
        # Velocity PID gains [vx, vy, vz]
        self.Kp_vel = np.array([1.0, 1.0, 1.0])
        self.Ki_vel = np.array([0.0, 0.0, 0.0])
        self.Kd_vel = np.array([0.1, 0.1, 0.1])
        
        # Attitude control gains [qx, qy, qz] (or roll, pitch, yaw)
        # P+PID structure: P term (direct) + PID term (feedback)
        self.Kp_att = np.array([5.0, 5.0, 1.0])  # P gain for direct feedforward
        self.Kp_att_pid = np.array([4.0, 4.0, 0.8])  # P gain for PID term
        self.Ki_att = np.array([0.0, 0.0, 0.0])  # I gain for PID term
        self.Kd_att = np.array([0.5, 0.5, 0.1])  # D gain for PID term
        
        # Angular velocity PID gains [p, q, r]
        self.Kp_omega = np.array([1.0, 1.0, 0.5])
        self.Ki_omega = np.array([0.0, 0.0, 0.0])
        self.Kd_omega = np.array([0.1, 0.1, 0.05])
        
        # Integral terms (for integral action)
        self.integral_pos = np.zeros(3)
        self.integral_vel = np.zeros(3)
        self.integral_att = np.zeros(3)
        self.integral_omega = np.zeros(3)
        
        # Previous errors (for derivative action)
        self.prev_error_pos = np.zeros(3)
        self.prev_error_vel = np.zeros(3)
        self.prev_error_att = np.zeros(3)
        self.prev_error_omega = np.zeros(3)
        
        # Previous attitude reference (for derivative calculation)
        self.prev_att_ref_total = np.zeros(3)
        self.prev_att = np.zeros(3)
        
        # Previous time (for derivative calculation)
        self.prev_time = None
    
    def reset(self):
        """Reset integral terms and previous errors."""
        self.integral_pos = np.zeros(3)
        self.integral_vel = np.zeros(3)
        self.integral_att = np.zeros(3)
        self.integral_omega = np.zeros(3)
        self.prev_error_pos = np.zeros(3)
        self.prev_error_vel = np.zeros(3)
        self.prev_error_att = np.zeros(3)
        self.prev_error_omega = np.zeros(3)
        self.prev_att_ref_total = np.zeros(3)
        self.prev_att = np.zeros(3)
        self.prev_time = None
    
    def compute_control(self, state: np.ndarray, state_ref: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute control input using PID control law.
        
        Args:
            state: Current state [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
            state_ref: Reference state
            dt: Time step (s)
            
        Returns:
            Control input [phi, theta, thrust, tau_r]
        """
        # Extract state components
        pos = state[0:3]
        vel = state[3:6]
        att = state[6:9]  # qx, qy, qz
        omega = state[9:12]
        
        # Extract reference components
        pos_ref = state_ref[0:3]
        vel_ref = state_ref[3:6]
        att_ref = state_ref[6:9]
        omega_ref = state_ref[9:12]
        
        # Compute errors
        error_pos = pos_ref - pos
        error_vel = vel_ref - vel
        error_att = att_ref - att
        error_omega = omega_ref - omega
        
        # Update integrals
        # Note: integral_att will be updated later using error_att_total (for P+PID structure)
        self.integral_pos += error_pos * dt
        self.integral_vel += error_vel * dt
        # self.integral_att += error_att * dt  # Moved to P+PID attitude control section
        self.integral_omega += error_omega * dt
        
        # Compute derivatives (if dt > 0)
        if dt > 0 and self.prev_time is not None:
            deriv_error_pos = (error_pos - self.prev_error_pos) / dt
            deriv_error_vel = (error_vel - self.prev_error_vel) / dt
            deriv_error_att = (error_att - self.prev_error_att) / dt
            deriv_error_omega = (error_omega - self.prev_error_omega) / dt
        else:
            deriv_error_pos = np.zeros(3)
            deriv_error_vel = np.zeros(3)
            deriv_error_att = np.zeros(3)
            deriv_error_omega = np.zeros(3)
        
        # PX4-style P+PID position control
        # P term: direct feedforward from position error
        vel_cmd_p = self.Kp_pos * error_pos
        
        # PID term: feedback from position error
        vel_cmd_pid = (self.Kp_pos_pid * error_pos + 
                       self.Ki_pos * self.integral_pos + 
                       self.Kd_pos * deriv_error_pos)
        
        # Total velocity command: P + PID
        vel_cmd = vel_cmd_p + vel_cmd_pid
        
        # Limit velocity command to prevent excessive commands
        max_vel_cmd = 5.0  # m/s, maximum velocity command
        vel_cmd = np.clip(vel_cmd, -max_vel_cmd, max_vel_cmd)
        
        # Add to velocity reference
        vel_ref_total = vel_ref + vel_cmd
        
        # Middle loop: Velocity PID -> attitude reference
        # Compute velocity error for horizontal control
        error_vel_total = vel_ref_total - vel
        
        # Velocity PID for horizontal motion
        vel_cmd_xy = (self.Kp_vel[0:2] * error_vel_total[0:2] + 
                      self.Ki_vel[0:2] * self.integral_vel[0:2] + 
                      self.Kd_vel[0:2] * deriv_error_vel[0:2])
        
        # For horizontal motion, we need attitude to generate horizontal acceleration
        # The required attitude is approximately: att ≈ vel_cmd / g (for small angles)
        # In ENU frame: 
        # - To accelerate in +X direction (East), need to tilt backward (negative pitch, -qy)
        # - To accelerate in +Y direction (North), need to tilt right (positive roll, +qx)
        # Limit velocity command to prevent excessive attitude commands
        max_vel_cmd_xy = 3.0  # m/s, maximum horizontal velocity command
        vel_cmd_xy_limited = np.clip(vel_cmd_xy, -max_vel_cmd_xy, max_vel_cmd_xy)
        
        att_cmd_xy = np.array([
            vel_cmd_xy_limited[1] / self.params.G,   # qx (roll) from vy (North) - positive roll for +Y
            -vel_cmd_xy_limited[0] / self.params.G,  # qy (pitch) from vx (East) - negative pitch for +X
            0.0  # qz (yaw) from angular velocity control
        ])
        
        # Limit attitude command to prevent excessive angles (saturate at ~15 degrees)
        max_attitude_angle = np.deg2rad(15.0)  # 15 degrees in radians
        att_cmd_xy = np.clip(att_cmd_xy, -max_attitude_angle, max_attitude_angle)
        
        # Total attitude reference: reference + horizontal control
        att_ref_total = att_ref + att_cmd_xy
        
        # PX4-style P+PID attitude control
        # Compute attitude error from total reference
        error_att_total = att_ref_total - att
        
        # Compute derivative of attitude error
        if dt > 0 and self.prev_time is not None:
            deriv_error_att_total = (error_att_total - (self.prev_att_ref_total - self.prev_att)) / dt
        else:
            deriv_error_att_total = np.zeros(3)
        
        # P term: direct feedforward from attitude error
        omega_cmd_p = self.Kp_att * error_att_total
        
        # PID term: feedback from attitude error
        # Update integral for attitude error (use error_att_total for integral)
        self.integral_att += error_att_total * dt
        omega_cmd_pid = (self.Kp_att_pid * error_att_total + 
                         self.Ki_att * self.integral_att + 
                         self.Kd_att * deriv_error_att_total)
        
        # Total angular velocity command: P + PID
        omega_cmd = omega_cmd_p + omega_cmd_pid
        
        # Limit angular velocity command to prevent excessive rates
        max_omega_cmd = 1.5  # rad/s, maximum angular velocity command (reduced for stability)
        omega_cmd = np.clip(omega_cmd, -max_omega_cmd, max_omega_cmd)
        
        omega_ref_total = omega_ref + omega_cmd
        
        # Inner-most loop: Angular velocity PID -> control input
        u_omega = (self.Kp_omega * error_omega + 
                   self.Ki_omega * self.integral_omega + 
                   self.Kd_omega * deriv_error_omega)
        
        # Limit angular velocity control output
        max_u_omega = 2.0  # rad/s, maximum angular velocity control output (reduced for stability)
        u_omega = np.clip(u_omega, -max_u_omega, max_u_omega)
        
        # Thrust from vertical velocity/position control
        # In ENU frame: Z-axis is positive upward
        # - z > 0 means above ground (e.g., z = 1.0 is 1m altitude)
        # - vz > 0 means upward velocity (climbing)
        # - If z_ref > z_current (error_pos[2] > 0), we need to go up -> need vz > 0
        # - To achieve upward motion, we need upward acceleration (az > 0 in ENU)
        # - From dynamics: accel = (thrust_world / mass) + gravity_world = thrust/mass - g
        # - In body frame: thrust_body[2] = +thrust * cos(phi) * cos(theta)
        # - When converted to world frame and attitude is level: thrust_world[2] = +thrust
        # - For upward acceleration: (thrust/mass) - g > 0, so thrust > mass*g
        # - IMPORTANT: If phi or theta are non-zero, effective vertical thrust is reduced
        # - Solution: Use current attitude to compensate
        
        thrust_base = self.params.MASS * self.params.G
        
        # Position error contribution: if z_ref > z_current (error_pos[2] > 0), need more thrust
        # Velocity error contribution: if vz_ref > vz_current (error_vel[2] > 0), need more thrust
        pos_error_contribution = self.Kp_pos[2] * error_pos[2] + \
                                  self.Ki_pos[2] * self.integral_pos[2] + \
                                  self.Kd_pos[2] * deriv_error_pos[2]
        
        vel_error_contribution = self.Kp_vel[2] * error_vel[2] + \
                                 self.Ki_vel[2] * self.integral_vel[2] + \
                                 self.Kd_vel[2] * deriv_error_vel[2]
        
        # Calculate base thrust command
        # When error > 0 (need up), this gives: thrust_cmd > base_thrust
        thrust_cmd = thrust_base + (pos_error_contribution + vel_error_contribution) * self.params.MASS
        
        # Limit thrust command to reasonable range (before attitude compensation)
        thrust_min = 0.2 * self.params.MASS * self.params.G  # Minimum 20% of weight
        thrust_max = 2.5 * self.params.MASS * self.params.G  # Maximum 250% of weight (reduced for stability)
        thrust_cmd = np.clip(thrust_cmd, thrust_min, thrust_max)
        
        # Compensate for current attitude: if phi or theta are non-zero, vertical thrust component is reduced
        # thrust_body[2] = +thrust * cos(phi) * cos(theta)
        # To maintain the same vertical component, we need: thrust_compensated = thrust / (cos(phi) * cos(theta))
        # Use current attitude from state (approximate: qx ≈ roll, qy ≈ pitch for small angles)
        # For small angles: phi ≈ 2*qx, theta ≈ 2*qy (from quaternion to Euler conversion)
        current_phi_approx = 2.0 * att[0]  # Approximate roll from qx
        current_theta_approx = 2.0 * att[1]  # Approximate pitch from qy
        
        # Limit angles for safety
        current_phi_approx = np.clip(current_phi_approx, -np.pi/3, np.pi/3)
        current_theta_approx = np.clip(current_theta_approx, -np.pi/3, np.pi/3)
        
        cos_phi = np.cos(current_phi_approx)
        cos_theta = np.cos(current_theta_approx)
        attitude_factor = cos_phi * cos_theta
        
        # Avoid division by zero or very small values
        if attitude_factor < 0.2:
            attitude_factor = 0.2
        
        # Compensate thrust to account for attitude
        # This ensures that the vertical component of thrust remains sufficient
        thrust_cmd = thrust_cmd / attitude_factor
        
        # Limit thrust again after attitude compensation to prevent excessive values
        thrust_max_final = 3.0 * self.params.MASS * self.params.G  # Absolute maximum
        thrust_cmd = np.clip(thrust_cmd, thrust_min, thrust_max_final)
        
        # Map to control inputs
        # phi (roll deflection) and theta (pitch deflection) from total attitude reference
        # Convert quaternion to Euler angles, then map to deflection angles
        # For small angles: roll ≈ 2*qx, pitch ≈ 2*qy
        # phi and theta are deflection angles, which should match the Euler angles for small angles
        from scipy.spatial.transform import Rotation
        qw_total = np.sqrt(np.clip(1.0 - np.sum(att_ref_total**2), 0.0, 1.0))
        q_total = np.array([qw_total, att_ref_total[0], att_ref_total[1], att_ref_total[2]])
        rot_total = Rotation.from_quat([q_total[1], q_total[2], q_total[3], q_total[0]])
        euler_total = rot_total.as_euler('ZYX', degrees=False)  # [yaw, pitch, roll]
        
        # Extract roll and pitch
        roll = euler_total[2]  # roll angle (affects Y-acceleration)
        pitch = euler_total[1]  # pitch angle (affects X-acceleration)
        
        # Based on linearize function:
        # - phi affects vx (X-acceleration) -> corresponds to pitch
        # - theta affects vy (Y-acceleration) -> corresponds to roll
        # Add angular velocity feedback for damping
        phi = pitch + u_omega[1] * 0.1  # phi from pitch (affects X)
        theta = roll + u_omega[0] * 0.1  # theta from roll (affects Y)
        
        # Limit deflection angles to prevent instability
        # Typical TVC systems limit deflection to ±15-20 degrees
        max_deflection_angle = np.deg2rad(15.0)  # 15 degrees in radians (more conservative)
        phi = np.clip(phi, -max_deflection_angle, max_deflection_angle)
        theta = np.clip(theta, -max_deflection_angle, max_deflection_angle)
        
        # Yaw torque from angular velocity control
        tau_r = u_omega[2] * self.params.I_ZZ
        
        # Limit yaw torque
        max_tau_r = 0.5  # Nm, maximum yaw torque
        tau_r = np.clip(tau_r, -max_tau_r, max_tau_r)
        
        # Update previous errors and states
        self.prev_error_pos = error_pos
        self.prev_error_vel = error_vel
        self.prev_error_att = error_att
        self.prev_error_omega = error_omega
        self.prev_att_ref_total = att_ref_total
        self.prev_att = att
        self.prev_time = dt
        
        u = np.array([phi, theta, thrust_cmd, tau_r])
        
        return u
    
    def set_gains(self, gains_dict: dict):
        """
        Set PID gains from dictionary.
        
        Args:
            gains_dict: Dictionary with keys like 'Kp_pos', 'Ki_pos', etc.
        """
        if 'Kp_pos' in gains_dict:
            self.Kp_pos = np.array(gains_dict['Kp_pos'])
        if 'Ki_pos' in gains_dict:
            self.Ki_pos = np.array(gains_dict['Ki_pos'])
        if 'Kd_pos' in gains_dict:
            self.Kd_pos = np.array(gains_dict['Kd_pos'])
        if 'Kp_vel' in gains_dict:
            self.Kp_vel = np.array(gains_dict['Kp_vel'])
        if 'Ki_vel' in gains_dict:
            self.Ki_vel = np.array(gains_dict['Ki_vel'])
        if 'Kd_vel' in gains_dict:
            self.Kd_vel = np.array(gains_dict['Kd_vel'])
        if 'Kp_att' in gains_dict:
            self.Kp_att = np.array(gains_dict['Kp_att'])
        if 'Ki_att' in gains_dict:
            self.Ki_att = np.array(gains_dict['Ki_att'])
        if 'Kd_att' in gains_dict:
            self.Kd_att = np.array(gains_dict['Kd_att'])
        if 'Kp_omega' in gains_dict:
            self.Kp_omega = np.array(gains_dict['Kp_omega'])
        if 'Ki_omega' in gains_dict:
            self.Ki_omega = np.array(gains_dict['Ki_omega'])
        if 'Kd_omega' in gains_dict:
            self.Kd_omega = np.array(gains_dict['Kd_omega'])
    
    def compute_attitude_control_only(self, state: np.ndarray, att_ref_euler: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute control input for attitude-only control (debugging mode).
        Only controls attitude, position and velocity are ignored.
        
        Args:
            state: Current state [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
            att_ref_euler: Reference attitude as Euler angles [roll, pitch, yaw] in radians
            dt: Time step (s)
            
        Returns:
            Control input [phi, theta, thrust, tau_r]
        """
        from scipy.spatial.transform import Rotation
        
        # Extract current attitude
        att = state[6:9]  # qx, qy, qz
        omega = state[9:12]  # p, q, r
        
        # Convert reference Euler angles to quaternion
        rot_ref = Rotation.from_euler('ZYX', [att_ref_euler[2], att_ref_euler[1], att_ref_euler[0]], degrees=False)
        quat_ref = rot_ref.as_quat()  # [x, y, z, w]
        att_ref = quat_ref[:3]  # qx, qy, qz (vector part)
        
        # Reference angular velocity (zero for attitude tracking)
        omega_ref = np.zeros(3)
        
        # Compute attitude error
        error_att = att_ref - att
        
        # Update integral for attitude error
        self.integral_att += error_att * dt
        
        # Compute derivative of attitude error
        if dt > 0 and self.prev_time is not None:
            deriv_error_att = (error_att - self.prev_error_att) / dt
        else:
            deriv_error_att = np.zeros(3)
        
        # PX4-style P+PID attitude control
        # P term: direct feedforward from attitude error
        omega_cmd_p = self.Kp_att * error_att
        
        # PID term: feedback from attitude error
        omega_cmd_pid = (self.Kp_att_pid * error_att + 
                         self.Ki_att * self.integral_att + 
                         self.Kd_att * deriv_error_att)
        
        # Total angular velocity command: P + PID
        omega_cmd = omega_cmd_p + omega_cmd_pid
        
        # Limit angular velocity command
        max_omega_cmd = 1.5  # rad/s
        omega_cmd = np.clip(omega_cmd, -max_omega_cmd, max_omega_cmd)
        
        # Total angular velocity reference: base reference + command
        omega_ref_total = omega_ref + omega_cmd
        
        # Compute angular velocity error (should use omega_ref_total, not omega_ref)
        error_omega = omega_ref_total - omega
        
        # Update integral for angular velocity error
        self.integral_omega += error_omega * dt
        
        # Compute derivative of angular velocity error
        if dt > 0 and self.prev_time is not None:
            deriv_error_omega = (error_omega - self.prev_error_omega) / dt
        else:
            deriv_error_omega = np.zeros(3)
        
        # Angular velocity PID -> control input
        u_omega = (self.Kp_omega * error_omega + 
                   self.Ki_omega * self.integral_omega + 
                   self.Kd_omega * deriv_error_omega)
        
        # Limit angular velocity control output
        max_u_omega = 2.0  # rad/s
        u_omega = np.clip(u_omega, -max_u_omega, max_u_omega)
        
        # For attitude-only control, use equilibrium thrust (hover)
        thrust_cmd = self.params.MASS * self.params.G
        
        # Map to control inputs
        # Based on linearize function:
        # - phi affects vx (X-acceleration) and p (roll rate) -> corresponds to pitch (qy)
        # - theta affects vy (Y-acceleration) and q (pitch rate) -> corresponds to roll (qx)
        # - u_omega[0] is control for p (roll rate) -> should affect theta
        # - u_omega[1] is control for q (pitch rate) -> should affect phi
        # - u_omega[2] is control for r (yaw rate) -> should affect tau_r
        
        # Use angular velocity control output as primary control input
        # The angular velocity PID already incorporates attitude error through omega_cmd
        # For small angles, we can directly use u_omega as control input
        # But we need to scale it appropriately to get deflection angles
        
        # Convert angular velocity control to deflection angles
        # u_omega has units of rad/s, but we need deflection angles in rad
        # Based on B matrix: B[9,0] = -l*m*g/I_XX for phi -> p
        # So: p_dot = B[9,0] * phi, which means phi ≈ p_dot / B[9,0]
        # But we want: phi ≈ u_omega[1] / (some scale factor)
        
        # Simpler approach: use a proportional mapping from angular velocity control
        # The angular velocity control u_omega already incorporates attitude error
        # Scale factor: convert rad/s control to rad deflection angle
        # Typical scale: 0.1-0.5 rad deflection per rad/s angular velocity command
        
        # Use angular velocity control output with appropriate scaling
        # Based on B matrix:
        # - phi affects p_dot (roll rate) and vx_dot (X-acceleration)
        # - theta affects q_dot (pitch rate) and vy_dot (Y-acceleration)
        # - B[9, 0] < 0: phi > 0 -> p_dot < 0 (negative sign)
        # - B[10, 1] < 0: theta > 0 -> q_dot < 0 (negative sign)
        # So we need to negate the control inputs to get correct direction:
        # - u_omega[0] controls p (roll rate) -> should map to -phi
        # - u_omega[1] controls q (pitch rate) -> should map to -theta
        scale_factor = 0.2  # Scale factor: rad deflection per rad/s control
        phi = -u_omega[0] * scale_factor  # phi from roll rate control (p), negated for correct sign
        theta = -u_omega[1] * scale_factor  # theta from pitch rate control (q), negated for correct sign
        
        # Limit deflection angles
        max_deflection_angle = np.deg2rad(15.0)  # 15 degrees
        phi = np.clip(phi, -max_deflection_angle, max_deflection_angle)
        theta = np.clip(theta, -max_deflection_angle, max_deflection_angle)
        
        # Yaw torque from angular velocity control
        tau_r = u_omega[2] * self.params.I_ZZ
        
        # Limit yaw torque
        max_tau_r = 0.5  # Nm
        tau_r = np.clip(tau_r, -max_tau_r, max_tau_r)
        
        # Update previous errors and states
        self.prev_error_att = error_att
        self.prev_error_omega = error_omega
        self.prev_att = att
        self.prev_time = dt
        
        u = np.array([phi, theta, thrust_cmd, tau_r])
        
        return u
