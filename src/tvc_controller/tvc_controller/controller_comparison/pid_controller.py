#!/usr/bin/env python3
"""
PID Controller for TVC Rocket (Strapdown Architecture)

This module implements a strapdown PID controller for the TVC rocket.
The controller is divided into two independent parts:
1. Position Control: Controls position and velocity
2. Attitude Control: Controls attitude and angular velocity

This strapdown architecture allows each part to work independently or together.
"""

import numpy as np
from typing import Optional, Tuple
import sys
import os

# Handle both relative and absolute imports
if __name__ == "__main__" or __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from controller_comparison.rocket_dynamics import PhyParams
else:
    from .rocket_dynamics import PhyParams


class PositionControl:
    """
    Strapdown Position Control Module
    
    Controls position and velocity independently.
    Uses PX4-style P+PID structure for position control.
    """
    
    def __init__(self, phy_params: PhyParams, max_deflection_angle_deg: float = 15.0):
        """Initialize position control module."""
        self.params = phy_params
        self.max_deflection_angle = np.deg2rad(max_deflection_angle_deg)
        
        # Position control gains [x, y, z] - only P term
        self.Kp_pos = np.array([1.0, 1.0, 10.0])
        
        # Velocity control gains [vx, vy, vz] - PID structure
        self.Kp_vel = np.array([1.0, 1.0, 1.0])
        self.Ki_vel = np.array([0.0, 0.0, 0.0])
        self.Kd_vel = np.array([0.1, 0.1, 0.1])
        
        # Integral terms (only for velocity control)
        self.integral_vel = np.zeros(3)
        
        # Previous errors (only for velocity control)
        self.prev_error_vel = np.zeros(3)
        self.prev_time = None
    
    def reset(self):
        """Reset integral terms and previous errors."""
        self.integral_vel = np.zeros(3)
        self.prev_error_vel = np.zeros(3)
        self.prev_time = None
    
    def compute(self, pos: np.ndarray, vel: np.ndarray, 
                pos_ref: np.ndarray, vel_ref: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute position control outputs.
        
        Args:
            pos: Current position [x, y, z]
            vel: Current velocity [vx, vy, vz]
            pos_ref: Reference position [x, y, z]
            vel_ref: Reference velocity [vx, vy, vz]
            dt: Time step (s)
            
        Returns:
            Tuple of (att_cmd_xy, thrust_cmd, vel_cmd_total)
            - att_cmd_xy: Attitude command for horizontal control [qx, qy, 0]
            - thrust_cmd: Thrust command (N)
            - vel_cmd_total: Total velocity command [vx, vy, vz]
        """
        # Compute errors
        error_pos = pos_ref - pos
        error_vel = vel_ref - vel
        
        # Update velocity integral
        self.integral_vel += error_vel * dt
        
        # Compute velocity derivative
        if dt > 0 and self.prev_time is not None:
            deriv_error_vel = (error_vel - self.prev_error_vel) / dt
        else:
            deriv_error_vel = np.zeros(3)
        
        # Position control: only P term
        vel_cmd = self.Kp_pos * error_pos
        
        # Limit velocity command
        max_vel_cmd = 5.0  # m/s
        vel_cmd = np.clip(vel_cmd, -max_vel_cmd, max_vel_cmd)
        
        # Total velocity command
        vel_cmd_total = vel_ref + vel_cmd
        
        # Velocity PID for horizontal control
        error_vel_total = vel_cmd_total - vel
        vel_cmd_xy = (self.Kp_vel[0:2] * error_vel_total[0:2] + 
                      self.Ki_vel[0:2] * self.integral_vel[0:2] + 
                      self.Kd_vel[0:2] * deriv_error_vel[0:2])
        
        # Limit horizontal velocity command
        max_vel_cmd_xy = 3.0  # m/s
        vel_cmd_xy_limited = np.clip(vel_cmd_xy, -max_vel_cmd_xy, max_vel_cmd_xy)
        
        # Convert to attitude command for horizontal control
        att_cmd_xy = np.array([
            vel_cmd_xy_limited[1] / self.params.G,   # qx (roll) from vy
            -vel_cmd_xy_limited[0] / self.params.G,  # qy (pitch) from vx
            0.0  # qz (yaw)
        ])
        
        # Limit attitude command
        att_cmd_xy = np.clip(att_cmd_xy, -self.max_deflection_angle, self.max_deflection_angle)
        
        # Thrust control from vertical position/velocity error
        # Position: only P term, Velocity: PID
        thrust_base = self.params.MASS * self.params.G
        pos_error_contribution = self.Kp_pos[2] * error_pos[2]
        # Velocity error contribution: PID
        vel_error_contribution = (self.Kp_vel[2] * error_vel[2] + 
                                  self.Ki_vel[2] * self.integral_vel[2] + 
                                  self.Kd_vel[2] * deriv_error_vel[2])
        
        thrust_cmd = thrust_base + (pos_error_contribution + vel_error_contribution) * self.params.MASS
        
        # Limit thrust
        thrust_min = 0.2 * self.params.MASS * self.params.G
        thrust_max = 2.5 * self.params.MASS * self.params.G
        thrust_cmd = np.clip(thrust_cmd, thrust_min, thrust_max)
        
        # Update previous errors
        self.prev_error_vel = error_vel
        self.prev_time = dt
        
        return att_cmd_xy, thrust_cmd, vel_cmd_total


class AttitudeControl:
    """
    Strapdown Attitude Control Module
    
    Controls attitude and angular velocity independently.
    Uses PX4-style P+PID structure for attitude control.
    """
    
    def __init__(self, phy_params: PhyParams, max_angular_velocity_deg_s: float = 86.0):
        """Initialize attitude control module."""
        self.params = phy_params
        self.max_angular_velocity = np.deg2rad(max_angular_velocity_deg_s)
        
        # Attitude control gains [qx, qy, qz] - only P term
        self.Kp_att = np.array([5.0, 5.0, 1.0])
        
        # Angular velocity control gains [p, q, r] - PID structure
        self.Kp_omega = np.array([1.0, 1.0, 0.5])
        self.Ki_omega = np.array([0.0, 0.0, 0.0])
        self.Kd_omega = np.array([0.1, 0.1, 0.05])
        
        # Integral terms (only for angular velocity control)
        self.integral_omega = np.zeros(3)
        
        # Previous errors (only for angular velocity control)
        self.prev_error_omega = np.zeros(3)
        self.prev_time = None
    
    def reset(self):
        """Reset integral terms and previous errors."""
        self.integral_omega = np.zeros(3)
        self.prev_error_omega = np.zeros(3)
        self.prev_time = None
    
    def compute(self, att: np.ndarray, omega: np.ndarray,
                att_ref: np.ndarray, omega_ref: np.ndarray, dt: float) -> Tuple[np.ndarray, float]:
        """
        Compute attitude control outputs.
        
        Args:
            att: Current attitude [qx, qy, qz]
            omega: Current angular velocity [p, q, r]
            att_ref: Reference attitude [qx, qy, qz]
            omega_ref: Reference angular velocity [p, q, r]
            dt: Time step (s)
            
        Returns:
            Tuple of (u_omega, tau_r)
            - u_omega: Angular velocity control output [p_cmd, q_cmd, r_cmd]
            - tau_r: Yaw torque (Nm)
        """
        # Compute attitude error
        error_att = att_ref - att
        
        # Attitude control: only P term
        omega_cmd = self.Kp_att * error_att
        
        # Limit angular velocity command
        omega_cmd = np.clip(omega_cmd, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Total angular velocity reference
        omega_ref_total = omega_ref + omega_cmd
        
        # Compute angular velocity error
        error_omega = omega_ref_total - omega
        
        # Update integral
        self.integral_omega += error_omega * dt
        
        # Compute derivative
        if dt > 0 and self.prev_time is not None:
            deriv_error_omega = (error_omega - self.prev_error_omega) / dt
        else:
            deriv_error_omega = np.zeros(3)
        
        # Angular velocity PID
        u_omega = (self.Kp_omega * error_omega + 
                   self.Ki_omega * self.integral_omega + 
                   self.Kd_omega * deriv_error_omega)
        
        # Limit angular velocity control output
        max_u_omega = 2.0  # rad/s
        u_omega = np.clip(u_omega, -max_u_omega, max_u_omega)
        
        # Yaw torque
        tau_r = u_omega[2] * self.params.I_ZZ
        max_tau_r = 0.5  # Nm
        tau_r = np.clip(tau_r, -max_tau_r, max_tau_r)
        
        # Update previous errors
        self.prev_error_omega = error_omega
        self.prev_time = dt
        
        return u_omega, tau_r


class PIDController:
    """
    Strapdown PID Controller for TVC Rocket.
    
    The controller consists of two independent strapdown modules:
    1. Position Control: Controls position and velocity
    2. Attitude Control: Controls attitude and angular velocity
    
    Uses cascaded control structure:
    - Position loop: P only -> velocity command
    - Velocity loop: PID -> attitude reference (for horizontal control) and thrust
    - Attitude loop: P only -> angular velocity command
    - Angular velocity loop: PID -> control input
    """
    
    def __init__(self, phy_params: PhyParams, 
                 max_deflection_angle_deg: float = 15.0,
                 max_angular_velocity_deg_s: float = 86.0):
        """
        Initialize PID controller.
        
        Args:
            phy_params: Physical parameters
            max_deflection_angle_deg: Maximum deflection angle in degrees (default: 15.0)
            max_angular_velocity_deg_s: Maximum angular velocity in deg/s (default: 86.0, ~1.5 rad/s)
        """
        self.params = phy_params
        
        # Control limits
        self.max_deflection_angle = np.deg2rad(max_deflection_angle_deg)
        self.max_angular_velocity = np.deg2rad(max_angular_velocity_deg_s)
        
        # Initialize strapdown control modules
        self.position_control = PositionControl(phy_params, max_deflection_angle_deg)
        self.attitude_control = AttitudeControl(phy_params, max_angular_velocity_deg_s)
        
        # Expose gains for backward compatibility
        self.Kp_pos = self.position_control.Kp_pos
        self.Kp_vel = self.position_control.Kp_vel
        self.Ki_vel = self.position_control.Ki_vel
        self.Kd_vel = self.position_control.Kd_vel
        self.Kp_att = self.attitude_control.Kp_att
        self.Kp_omega = self.attitude_control.Kp_omega
        self.Ki_omega = self.attitude_control.Ki_omega
        self.Kd_omega = self.attitude_control.Kd_omega
    
    def reset(self):
        """Reset integral terms and previous errors."""
        self.position_control.reset()
        self.attitude_control.reset()
    
    def compute_control(self, state: np.ndarray, state_ref: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute control input using strapdown PID control law.
        
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
        
        # Strapdown Position Control
        att_cmd_xy, thrust_cmd, vel_cmd_total = self.position_control.compute(
            pos, vel, pos_ref, vel_ref, dt
        )
        
        # Combine attitude reference: base reference + horizontal control command
        att_ref_total = att_ref + att_cmd_xy
        
        # Compensate thrust for current attitude
        current_phi_approx = 2.0 * att[0]
        current_theta_approx = 2.0 * att[1]
        current_phi_approx = np.clip(current_phi_approx, -np.pi/3, np.pi/3)
        current_theta_approx = np.clip(current_theta_approx, -np.pi/3, np.pi/3)
        cos_phi = np.cos(current_phi_approx)
        cos_theta = np.cos(current_theta_approx)
        attitude_factor = cos_phi * cos_theta
        if attitude_factor < 0.2:
            attitude_factor = 0.2
        thrust_cmd = thrust_cmd / attitude_factor
        
        # Limit thrust after attitude compensation
        thrust_min = 0.2 * self.params.MASS * self.params.G
        thrust_max_final = 3.0 * self.params.MASS * self.params.G
        thrust_cmd = np.clip(thrust_cmd, thrust_min, thrust_max_final)
        
        # Strapdown Attitude Control
        u_omega, tau_r = self.attitude_control.compute(
            att, omega, att_ref_total, omega_ref, dt
        )
        
        # Map to control inputs
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
        
        # Limit deflection angles
        max_deflection_angle = np.deg2rad(15.0)
        phi = np.clip(phi, -max_deflection_angle, max_deflection_angle)
        theta = np.clip(theta, -max_deflection_angle, max_deflection_angle)
        
        u = np.array([phi, theta, thrust_cmd, tau_r])
        return u
    
    def set_gains(self, gains_dict: dict):
        """
        Set PID gains from dictionary.
        Updates both strapdown control modules.
        
        Args:
            gains_dict: Dictionary with keys like 'Kp_pos', 'Ki_pos', etc.
        """
        # Update position control gains (only P term)
        if 'Kp_pos' in gains_dict:
            self.position_control.Kp_pos = np.array(gains_dict['Kp_pos'])
            self.Kp_pos = self.position_control.Kp_pos
        if 'Kp_vel' in gains_dict:
            self.position_control.Kp_vel = np.array(gains_dict['Kp_vel'])
            self.Kp_vel = self.position_control.Kp_vel
        if 'Kp_vel_pid' in gains_dict:
            self.position_control.Kp_vel_pid = np.array(gains_dict['Kp_vel_pid'])
            self.Kp_vel_pid = self.position_control.Kp_vel_pid
        if 'Ki_vel' in gains_dict:
            self.position_control.Ki_vel = np.array(gains_dict['Ki_vel'])
            self.Ki_vel = self.position_control.Ki_vel
        if 'Kd_vel' in gains_dict:
            self.position_control.Kd_vel = np.array(gains_dict['Kd_vel'])
            self.Kd_vel = self.position_control.Kd_vel
        
        # Update attitude control gains (only P term)
        if 'Kp_att' in gains_dict:
            self.attitude_control.Kp_att = np.array(gains_dict['Kp_att'])
            self.Kp_att = self.attitude_control.Kp_att
        if 'Kp_omega' in gains_dict:
            self.attitude_control.Kp_omega = np.array(gains_dict['Kp_omega'])
            self.Kp_omega = self.attitude_control.Kp_omega
        if 'Ki_omega' in gains_dict:
            self.attitude_control.Ki_omega = np.array(gains_dict['Ki_omega'])
            self.Ki_omega = self.attitude_control.Ki_omega
        if 'Kd_omega' in gains_dict:
            self.attitude_control.Kd_omega = np.array(gains_dict['Kd_omega'])
            self.Kd_omega = self.attitude_control.Kd_omega
    
    def set_limits(self, max_deflection_angle_deg: float = None, max_angular_velocity_deg_s: float = None):
        """
        Set control limits.
        
        Args:
            max_deflection_angle_deg: Maximum deflection angle in degrees
            max_angular_velocity_deg_s: Maximum angular velocity in deg/s
        """
        if max_deflection_angle_deg is not None:
            self.max_deflection_angle = np.deg2rad(max_deflection_angle_deg)
            self.position_control.max_deflection_angle = self.max_deflection_angle
        if max_angular_velocity_deg_s is not None:
            self.max_angular_velocity = np.deg2rad(max_angular_velocity_deg_s)
            self.attitude_control.max_angular_velocity = self.max_angular_velocity
    
    def compute_attitude_control_only(self, state: np.ndarray, att_ref_euler: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute control input for attitude-only control (debugging mode).
        Only controls attitude, position and velocity are ignored.
        Uses strapdown attitude control module.
        
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
        
        # Use strapdown attitude control
        u_omega, tau_r = self.attitude_control.compute(att, omega, att_ref, omega_ref, dt)
        
        # For attitude-only control, use equilibrium thrust (hover)
        thrust_cmd = self.params.MASS * self.params.G
        
        # Map to control inputs
        scale_factor = 0.2  # Scale factor: rad deflection per rad/s control
        phi = -u_omega[0] * scale_factor  # phi from roll rate control (p)
        theta = -u_omega[1] * scale_factor  # theta from pitch rate control (q)
        
        # Limit deflection angles
        phi = np.clip(phi, -self.max_deflection_angle, self.max_deflection_angle)
        theta = np.clip(theta, -self.max_deflection_angle, self.max_deflection_angle)
        
        u = np.array([phi, theta, thrust_cmd, tau_r])
        return u


# ============================================================================
# Test Functions (from pid_attitude_test_debug.py)
# ============================================================================

def quaternion_to_euler(q_vec: np.ndarray) -> np.ndarray:
    """Convert quaternion vector to Euler angles [roll, pitch, yaw] in radians."""
    from scipy.spatial.transform import Rotation
    qw = np.sqrt(np.clip(1.0 - np.sum(q_vec**2), 0.0, 1.0))
    quat = np.array([qw, q_vec[0], q_vec[1], q_vec[2]])
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler = rot.as_euler('ZYX', degrees=False)  # [yaw, pitch, roll]
    return np.array([euler[2], euler[1], euler[0]])  # [roll, pitch, yaw]


def test_multiple_steps():
    """Test multiple control steps to see the response."""
    import matplotlib.pyplot as plt
    
    if __name__ == "__main__" or __package__ is None:
        from controller_comparison.rocket_dynamics import RocketDynamics
    else:
        from .rocket_dynamics import RocketDynamics
    
    print("\n" + "=" * 80)
    print("PID Attitude Control - Multiple Steps Test")
    print("=" * 80)
    
    phy_params = PhyParams(
        MASS=0.6570,
        G=9.81,
        I_XX=0.001,
        I_YY=0.001,
        I_ZZ=0.0001,
        DIST_COM_2_THRUST=0.1
    )
    
    pid = PIDController(phy_params)
    dynamics = RocketDynamics(phy_params)
    
    state = np.array([
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])
    
    att_ref_euler = np.deg2rad(np.array([10.0, 0.0, 0.0]))
    dt = 0.01
    sim_time = 3.0
    num_steps = int(sim_time / dt)
    
    pid.reset()
    
    time_points = []
    att_euler_traj = []
    att_ref_euler_traj = []
    omega_traj = []
    control_traj = []
    
    print("\nTesting multiple control steps...")
    print(f"{'Step':<6} {'Roll (deg)':<12} {'Pitch (deg)':<12} {'Yaw (deg)':<12} {'p (deg/s)':<12} {'phi (deg)':<12} {'theta (deg)':<12}")
    print("-" * 80)
    
    for step in range(num_steps):
        t = step * dt
        u = pid.compute_attitude_control_only(state, att_ref_euler, dt)
        state_dot = dynamics.compute_dynamics(t, state, u)
        state = state + state_dot * dt
        att_euler = quaternion_to_euler(state[6:9])
        
        time_points.append(t)
        att_euler_traj.append(np.rad2deg(att_euler))
        att_ref_euler_traj.append(np.rad2deg(att_ref_euler))
        omega_traj.append(np.rad2deg(state[9:12]))
        control_traj.append(u.copy())
        
        if step % 10 == 0 or step < 10:
            print(f"{step+1:<6} {att_euler_traj[-1][0]:<12.4f} {att_euler_traj[-1][1]:<12.4f} "
                  f"{att_euler_traj[-1][2]:<12.4f} {omega_traj[-1][0]:<12.4f} "
                  f"{np.rad2deg(u[0]):<12.4f} {np.rad2deg(u[1]):<12.4f}")
    
    time_points = np.array(time_points)
    att_euler_traj = np.array(att_euler_traj)
    att_ref_euler_traj = np.array(att_ref_euler_traj)
    omega_traj = np.array(omega_traj)
    control_traj = np.array(control_traj)
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'PID Attitude Control Response\nReference: Roll={np.rad2deg(att_ref_euler[0]):.1f}°, '
                 f'Pitch={np.rad2deg(att_ref_euler[1]):.1f}°, Yaw={np.rad2deg(att_ref_euler[2]):.1f}°', fontsize=14)
    
    axes[0, 0].plot(time_points, att_euler_traj[:, 0], 'b-', label='Roll', linewidth=2)
    axes[0, 0].plot(time_points, att_ref_euler_traj[:, 0], 'b--', label='Roll Ref', linewidth=1)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Roll (deg)')
    axes[0, 0].set_title('Control Result: Roll')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].plot(time_points, att_euler_traj[:, 1], 'r-', label='Pitch', linewidth=2)
    axes[0, 1].plot(time_points, att_ref_euler_traj[:, 1], 'r--', label='Pitch Ref', linewidth=1)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Pitch (deg)')
    axes[0, 1].set_title('Control Result: Pitch')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    axes[0, 2].plot(time_points, att_euler_traj[:, 2], 'g-', label='Yaw', linewidth=2)
    axes[0, 2].plot(time_points, att_ref_euler_traj[:, 2], 'g--', label='Yaw Ref', linewidth=1)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Yaw (deg)')
    axes[0, 2].set_title('Control Result: Yaw')
    axes[0, 2].grid(True)
    axes[0, 2].legend()
    
    axes[1, 0].plot(time_points, omega_traj[:, 0], 'b-', label='p (roll rate)', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('p (deg/s)')
    axes[1, 0].set_title('Angular Velocity: p (Roll Rate)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    axes[1, 1].plot(time_points, omega_traj[:, 1], 'r-', label='q (pitch rate)', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('q (deg/s)')
    axes[1, 1].set_title('Angular Velocity: q (Pitch Rate)')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    axes[1, 2].plot(time_points, omega_traj[:, 2], 'g-', label='r (yaw rate)', linewidth=2)
    axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('r (deg/s)')
    axes[1, 2].set_title('Angular Velocity: r (Yaw Rate)')
    axes[1, 2].grid(True)
    axes[1, 2].legend()
    
    axes[2, 0].plot(time_points, np.rad2deg(control_traj[:, 0]), 'b-', label='Phi', linewidth=2)
    axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Phi (deg)')
    axes[2, 0].set_title('Control Input: Phi (Deflection)')
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    
    axes[2, 1].plot(time_points, np.rad2deg(control_traj[:, 1]), 'r-', label='Theta', linewidth=2)
    axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Theta (deg)')
    axes[2, 1].set_title('Control Input: Theta (Deflection)')
    axes[2, 1].grid(True)
    axes[2, 1].legend()
    
    axes[2, 2].plot(time_points, control_traj[:, 2], 'g-', label='Thrust', linewidth=2)
    axes[2, 2].axhline(y=phy_params.MASS * phy_params.G, color='r', linestyle='--', alpha=0.7, label='Equilibrium')
    axes[2, 2].set_xlabel('Time (s)')
    axes[2, 2].set_ylabel('Thrust (N)')
    axes[2, 2].set_title('Control Input: Thrust')
    axes[2, 2].grid(True)
    axes[2, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run tests
    test_multiple_steps()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
