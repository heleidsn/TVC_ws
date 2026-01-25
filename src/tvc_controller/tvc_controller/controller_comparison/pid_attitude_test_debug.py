#!/usr/bin/env python3
"""
PID Attitude Control Detailed Debug Tool

This script provides detailed debugging output for PID attitude control,
showing all intermediate values in the control loop.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import sys
import os

# Handle imports
if __name__ == "__main__" or __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from controller_comparison.rocket_dynamics import RocketDynamics, PhyParams
    from controller_comparison.pid_controller import PIDController
else:
    from .rocket_dynamics import RocketDynamics, PhyParams
    from .pid_controller import PIDController


def quaternion_to_euler(q_vec: np.ndarray) -> np.ndarray:
    """Convert quaternion vector to Euler angles [roll, pitch, yaw] in degrees."""
    qw = np.sqrt(np.clip(1.0 - np.sum(q_vec**2), 0.0, 1.0))
    quat = np.array([qw, q_vec[0], q_vec[1], q_vec[2]])
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler = rot.as_euler('ZYX', degrees=False)  # [yaw, pitch, roll]
    return np.array([euler[2], euler[1], euler[0]])  # [roll, pitch, yaw]


def test_single_step():
    """Test a single control step with detailed output."""
    print("=" * 80)
    print("PID Attitude Control - Single Step Debug")
    print("=" * 80)
    
    # Physical parameters
    phy_params = PhyParams(
        MASS=0.6570,
        G=9.81,
        I_XX=0.001,
        I_YY=0.001,
        I_ZZ=0.0001,
        DIST_COM_2_THRUST=0.1
    )
    
    # Initialize controller and dynamics
    pid = PIDController(phy_params)
    dynamics = RocketDynamics(phy_params)
    
    # Initial state: at origin, zero attitude, zero angular velocity
    state = np.array([
        0.0, 0.0, 1.0,  # Position: x, y, z (1m altitude)
        0.0, 0.0, 0.0,  # Velocity: vx, vy, vz
        0.0, 0.0, 0.0,  # Attitude: qx, qy, qz (zero attitude)
        0.0, 0.0, 0.0   # Angular velocity: p, q, r
    ])
    
    # Reference attitude: Roll = 10 degrees
    att_ref_euler = np.deg2rad(np.array([10.0, 0.0, 0.0]))
    dt = 0.01
    
    # Reset controller
    pid.reset()
    
    print("\n1. Initial State:")
    print(f"   Position: {state[0:3]}")
    print(f"   Velocity: {state[3:6]}")
    print(f"   Attitude (qx, qy, qz): {state[6:9]}")
    print(f"   Angular velocity (p, q, r): {state[9:12]}")
    
    print("\n2. Reference Attitude:")
    print(f"   Roll: {np.rad2deg(att_ref_euler[0]):.2f}°")
    print(f"   Pitch: {np.rad2deg(att_ref_euler[1]):.2f}°")
    print(f"   Yaw: {np.rad2deg(att_ref_euler[2]):.2f}°")
    
    # Convert reference to quaternion
    rot_ref = Rotation.from_euler('ZYX', [att_ref_euler[2], att_ref_euler[1], att_ref_euler[0]], degrees=False)
    quat_ref = rot_ref.as_quat()
    att_ref = quat_ref[:3]
    print(f"   Reference quaternion (qx, qy, qz): {att_ref}")
    
    # Compute attitude error manually
    error_att_manual = att_ref - state[6:9]
    print("\n3. Attitude Error (manual calculation):")
    print(f"   error_qx: {error_att_manual[0]:.6f}")
    print(f"   error_qy: {error_att_manual[1]:.6f}")
    print(f"   error_qz: {error_att_manual[2]:.6f}")
    
    # Compute control
    print("\n4. Computing Control Input...")
    u = pid.compute_attitude_control_only(state, att_ref_euler, dt)
    
    print("\n5. Control Input:")
    print(f"   phi (deflection): {np.rad2deg(u[0]):.6f}°")
    print(f"   theta (deflection): {np.rad2deg(u[1]):.6f}°")
    print(f"   thrust: {u[2]:.6f} N")
    print(f"   tau_r: {u[3]:.6f} Nm")
    
    # Check internal controller state
    print("\n6. Controller Internal State:")
    print(f"   prev_error_att: {pid.prev_error_att}")
    print(f"   prev_error_omega: {pid.prev_error_omega}")
    print(f"   integral_att: {pid.integral_att}")
    print(f"   integral_omega: {pid.integral_omega}")
    
    # Compute dynamics
    print("\n7. Computing Dynamics...")
    state_dot = dynamics.compute_dynamics(0.0, state, u)
    
    print("\n8. State Derivatives:")
    print(f"   Position derivative: {state_dot[0:3]}")
    print(f"   Velocity derivative: {state_dot[3:6]}")
    print(f"   Attitude derivative: {state_dot[6:9]}")
    print(f"   Angular velocity derivative: {state_dot[9:12]}")
    print(f"   Angular velocity derivative (deg/s): {np.rad2deg(state_dot[9:12])}")
    
    # Check if control is working
    print("\n9. Control Effectiveness Check:")
    print(f"   Expected: Roll should increase (p_dot should be positive)")
    print(f"   Actual p_dot: {state_dot[9]:.6f} rad/s ({np.rad2deg(state_dot[9]):.2f} deg/s)")
    if state_dot[9] > 0:
        print("   ✓ Roll rate is increasing (correct direction)")
    else:
        print("   ✗ Roll rate is not increasing (wrong direction or too small)")
    
    # Plot single step visualization - simplified to one figure
    print("\n10. Generating visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Single Step Control Analysis', fontsize=14)
    
    # Plot 1: Attitude error (control result)
    error_euler = quaternion_to_euler(pid.prev_error_att)
    axes[0].bar(['Roll Error', 'Pitch Error', 'Yaw Error'], 
                np.rad2deg(error_euler), color=['b', 'r', 'g'], alpha=0.7)
    axes[0].set_ylabel('Attitude Error (deg)')
    axes[0].set_title('Control Result: Attitude Error')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Control inputs
    axes[1].bar(['phi (deg)', 'theta (deg)', 'thrust (N)', 'tau_r (Nm)'], 
                [np.rad2deg(u[0]), np.rad2deg(u[1]), u[2], u[3]], 
                color=['b', 'r', 'g', 'orange'], alpha=0.7)
    axes[1].set_ylabel('Control Input')
    axes[1].set_title('Control Inputs')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)


def test_multiple_steps():
    """Test multiple control steps to see the response."""
    print("\n" + "=" * 80)
    print("PID Attitude Control - Multiple Steps Test")
    print("=" * 80)
    
    # Physical parameters
    phy_params = PhyParams(
        MASS=0.6570,
        G=9.81,
        I_XX=0.001,
        I_YY=0.001,
        I_ZZ=0.0001,
        DIST_COM_2_THRUST=0.1
    )
    
    # Initialize controller and dynamics
    pid = PIDController(phy_params)
    dynamics = RocketDynamics(phy_params)
    
    # Initial state
    state = np.array([
        0.0, 0.0, 1.0,  # Position
        0.0, 0.0, 0.0,  # Velocity
        0.0, 0.0, 0.0,  # Attitude
        0.0, 0.0, 0.0   # Angular velocity
    ])
    
    # Reference attitude: Roll = 10 degrees
    att_ref_euler = np.deg2rad(np.array([10.0, 0.0, 0.0]))
    dt = 0.01
    sim_time = 3.0
    num_steps = int(sim_time / dt)
    
    # Reset controller
    pid.reset()
    
    # Storage for plotting
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
        
        # Compute control
        u = pid.compute_attitude_control_only(state, att_ref_euler, dt)

        # u = np.array([-0.05, 0.0, phy_params.MASS * phy_params.G, 0.0])
        
        # Compute dynamics
        state_dot = dynamics.compute_dynamics(t, state, u)
        
        # Integrate
        state = state + state_dot * dt
        
        # Convert to Euler angles
        att_euler = quaternion_to_euler(state[6:9])
        
        # Store for plotting
        time_points.append(t)
        att_euler_traj.append(np.rad2deg(att_euler))
        att_ref_euler_traj.append(np.rad2deg(att_ref_euler))
        omega_traj.append(np.rad2deg(state[9:12]))
        control_traj.append(u.copy())
        
        # Print every 10 steps
        if step % 10 == 0 or step < 10:
            print(f"{step+1:<6} {att_euler_traj[-1][0]:<12.4f} {att_euler_traj[-1][1]:<12.4f} "
                  f"{att_euler_traj[-1][2]:<12.4f} {omega_traj[-1][0]:<12.4f} "
                  f"{np.rad2deg(u[0]):<12.4f} {np.rad2deg(u[1]):<12.4f}")
    
    # Convert to numpy arrays
    time_points = np.array(time_points)
    att_euler_traj = np.array(att_euler_traj)
    att_ref_euler_traj = np.array(att_ref_euler_traj)
    omega_traj = np.array(omega_traj)
    control_traj = np.array(control_traj)
    
    # Plot results - separate plots for each metric (including angular velocity)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'PID Attitude Control Response\nReference: Roll={np.rad2deg(att_ref_euler[0]):.1f}°, '
                 f'Pitch={np.rad2deg(att_ref_euler[1]):.1f}°, Yaw={np.rad2deg(att_ref_euler[2]):.1f}°', fontsize=14)
    
    # Plot 1: Roll (control result)
    axes[0, 0].plot(time_points, att_euler_traj[:, 0], 'b-', label='Roll', linewidth=2)
    axes[0, 0].plot(time_points, att_ref_euler_traj[:, 0], 'b--', label='Roll Ref', linewidth=1)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Roll (deg)')
    axes[0, 0].set_title('Control Result: Roll')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # Plot 2: Pitch (control result)
    axes[0, 1].plot(time_points, att_euler_traj[:, 1], 'r-', label='Pitch', linewidth=2)
    axes[0, 1].plot(time_points, att_ref_euler_traj[:, 1], 'r--', label='Pitch Ref', linewidth=1)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Pitch (deg)')
    axes[0, 1].set_title('Control Result: Pitch')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # Plot 3: Yaw (control result)
    axes[0, 2].plot(time_points, att_euler_traj[:, 2], 'g-', label='Yaw', linewidth=2)
    axes[0, 2].plot(time_points, att_ref_euler_traj[:, 2], 'g--', label='Yaw Ref', linewidth=1)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Yaw (deg)')
    axes[0, 2].set_title('Control Result: Yaw')
    axes[0, 2].grid(True)
    axes[0, 2].legend()
    
    # Plot 4: Angular velocity p (roll rate)
    axes[1, 0].plot(time_points, omega_traj[:, 0], 'b-', label='p (roll rate)', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('p (deg/s)')
    axes[1, 0].set_title('Angular Velocity: p (Roll Rate)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # Plot 5: Angular velocity q (pitch rate)
    axes[1, 1].plot(time_points, omega_traj[:, 1], 'r-', label='q (pitch rate)', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('q (deg/s)')
    axes[1, 1].set_title('Angular Velocity: q (Pitch Rate)')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    # Plot 6: Angular velocity r (yaw rate)
    axes[1, 2].plot(time_points, omega_traj[:, 2], 'g-', label='r (yaw rate)', linewidth=2)
    axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('r (deg/s)')
    axes[1, 2].set_title('Angular Velocity: r (Yaw Rate)')
    axes[1, 2].grid(True)
    axes[1, 2].legend()
    
    # Plot 7: Phi (control input)
    axes[2, 0].plot(time_points, np.rad2deg(control_traj[:, 0]), 'b-', label='Phi', linewidth=2)
    axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Phi (deg)')
    axes[2, 0].set_title('Control Input: Phi (Deflection)')
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    
    # Plot 8: Theta (control input)
    axes[2, 1].plot(time_points, np.rad2deg(control_traj[:, 1]), 'r-', label='Theta', linewidth=2)
    axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Theta (deg)')
    axes[2, 1].set_title('Control Input: Theta (Deflection)')
    axes[2, 1].grid(True)
    axes[2, 1].legend()
    
    # Plot 9: Thrust (control input)
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


def test_control_gains():
    """Test different control gains to see their effect."""
    print("\n" + "=" * 80)
    print("PID Attitude Control - Gain Analysis")
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
    
    print("\nCurrent Gains:")
    print(f"   Kp_att: {pid.Kp_att}")
    print(f"   Kp_att_pid: {pid.Kp_att_pid}")
    print(f"   Ki_att: {pid.Ki_att}")
    print(f"   Kd_att: {pid.Kd_att}")
    print(f"   Kp_omega: {pid.Kp_omega}")
    print(f"   Ki_omega: {pid.Ki_omega}")
    print(f"   Kd_omega: {pid.Kd_omega}")
    
    # Test with a simple error
    error_att = np.array([0.1, 0.0, 0.0])  # Roll error
    error_omega = np.array([0.0, 0.0, 0.0])  # No angular velocity error initially
    
    print("\nTest with error_att = [0.1, 0, 0] (roll error):")
    
    # Compute omega_cmd
    omega_cmd_p = pid.Kp_att * error_att
    omega_cmd_pid = pid.Kp_att_pid * error_att  # No integral/derivative for first step
    omega_cmd = omega_cmd_p + omega_cmd_pid
    
    print(f"   omega_cmd_p: {omega_cmd_p}")
    print(f"   omega_cmd_pid: {omega_cmd_pid}")
    print(f"   omega_cmd: {omega_cmd}")
    
    # With omega_cmd, compute error_omega
    omega_ref_total = omega_cmd  # omega_ref = 0
    error_omega = omega_ref_total - np.zeros(3)  # omega = 0
    
    print(f"   omega_ref_total: {omega_ref_total}")
    print(f"   error_omega: {error_omega}")
    
    # Compute u_omega
    u_omega = pid.Kp_omega * error_omega  # No integral/derivative for first step
    
    print(f"   u_omega: {u_omega}")
    
    # Map to control inputs
    scale_factor = 0.2
    phi = -u_omega[0] * scale_factor  # Fixed: phi from u_omega[0]
    theta = -u_omega[1] * scale_factor  # Fixed: theta from u_omega[1]
    
    print(f"   phi (from u_omega[0]): {np.rad2deg(phi):.4f}°")
    print(f"   theta (from u_omega[1]): {np.rad2deg(theta):.4f}°")
    
    # Test with different error magnitudes for plotting
    error_magnitudes = [0.01, 0.05, 0.1, 0.2]
    omega_cmd_list = []
    u_omega_list = []
    control_list = []
    
    print("\nTesting different error magnitudes:")
    for error_mag in error_magnitudes:
        error_att_test = np.array([error_mag, 0.0, 0.0])
        omega_cmd_p_test = pid.Kp_att * error_att_test
        omega_cmd_pid_test = pid.Kp_att_pid * error_att_test
        omega_cmd_test = omega_cmd_p_test + omega_cmd_pid_test
        error_omega_test = omega_cmd_test - np.zeros(3)
        u_omega_test = pid.Kp_omega * error_omega_test
        phi_test = -u_omega_test[0] * scale_factor
        
        omega_cmd_list.append(omega_cmd_test[0])
        u_omega_list.append(u_omega_test[0])
        control_list.append(phi_test)
    
    # Plot gain analysis - simplified to one figure
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle('Control Gain Analysis', fontsize=14)
    
    error_rad = np.array(error_magnitudes)
    
    # Plot control input vs error
    axes.plot(error_rad, np.rad2deg(control_list), 'go-', linewidth=2, markersize=8, label='Control Input phi')
    axes.set_xlabel('Attitude Error (rad)')
    axes.set_ylabel('Control Input phi (deg)')
    axes.set_title('Control Result: Error -> Control Input')
    axes.grid(True)
    axes.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nExpected behavior:")
    print("   - Roll error (error_qx > 0) should produce phi < 0 (negative)")
    print("   - phi < 0 should produce roll rate p > 0 (positive)")
    print("   - p > 0 should reduce roll error")
    
    print("\n" + "=" * 80)


def test_dynamics_response():
    """Test the dynamics response to control inputs."""
    print("\n" + "=" * 80)
    print("PID Attitude Control - Dynamics Response Test")
    print("=" * 80)
    
    phy_params = PhyParams(
        MASS=0.6570,
        G=9.81,
        I_XX=0.001,
        I_YY=0.001,
        I_ZZ=0.0001,
        DIST_COM_2_THRUST=0.1
    )
    
    dynamics = RocketDynamics(phy_params)
    
    # Test state
    state = np.array([
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])
    
    print("\nTesting different control inputs:")
    print(f"{'phi (deg)':<12} {'theta (deg)':<12} {'p_dot (deg/s²)':<15} {'q_dot (deg/s²)':<15}")
    print("-" * 60)
    
    test_inputs = [
        (0.0, 0.0),
        (5.0, 0.0),
        (0.0, 5.0),
        (5.0, 5.0),
    ]
    
    phi_list = []
    theta_list = []
    p_dot_list = []
    q_dot_list = []
    
    for phi_deg, theta_deg in test_inputs:
        phi = np.deg2rad(phi_deg)
        theta = np.deg2rad(theta_deg)
        u = np.array([phi, theta, phy_params.MASS * phy_params.G, 0.0])
        
        state_dot = dynamics.compute_dynamics(0.0, state, u)
        
        phi_list.append(phi_deg)
        theta_list.append(theta_deg)
        p_dot_list.append(np.rad2deg(state_dot[9]))
        q_dot_list.append(np.rad2deg(state_dot[10]))
        
        print(f"{phi_deg:<12.1f} {theta_deg:<12.1f} {p_dot_list[-1]:<15.4f} {q_dot_list[-1]:<15.4f}")
    
    # Plot dynamics response - simplified to one figure
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle('Dynamics Response to Control Inputs', fontsize=14)
    
    # Plot control input vs response
    axes.plot(phi_list, p_dot_list, 'bo-', label='p_dot vs phi', linewidth=2, markersize=8)
    axes.plot(theta_list, q_dot_list, 'ro-', label='q_dot vs theta', linewidth=2, markersize=8)
    axes.set_xlabel('Control Input (deg)')
    axes.set_ylabel('Angular Acceleration (deg/s²)')
    axes.set_title('Control Result: Input -> Response')
    axes.grid(True)
    axes.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run all tests
    # test_single_step()
    test_multiple_steps()
    # test_control_gains()
    # test_dynamics_response()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
