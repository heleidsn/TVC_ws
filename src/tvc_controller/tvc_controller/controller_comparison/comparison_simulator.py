#!/usr/bin/env python3
"""
Controller Comparison Simulator

This module provides a simulation framework to compare different controllers:
- Full-state LQR
- Attitude-only LQR
- PID

All controllers are tested on the same rocket dynamics model.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import sys
import os

# Handle both relative and absolute imports
if __name__ == "__main__" or __package__ is None:
    # Running as script, add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from controller_comparison.rocket_dynamics import RocketDynamics, PhyParams
    from controller_comparison.lqr_full_state import LQRFullStateController
    from controller_comparison.lqr_attitude_only import LQRAttitudeOnlyController
    from controller_comparison.pid_controller import PIDController
else:
    # Running as module
    from .rocket_dynamics import RocketDynamics, PhyParams
    from .lqr_full_state import LQRFullStateController
    from .lqr_attitude_only import LQRAttitudeOnlyController
    from .pid_controller import PIDController


def quaternion_to_euler(q_vec: np.ndarray) -> np.ndarray:
    """
    Convert quaternion vector part to Euler angles (roll, pitch, yaw).
    
    Args:
        q_vec: Quaternion vector part [qx, qy, qz] (shape: (N, 3) or (3,))
        
    Returns:
        Euler angles [roll, pitch, yaw] in radians (shape: (N, 3) or (3,))
    """
    # Handle both single quaternion and array of quaternions
    single = False
    if q_vec.ndim == 1:
        single = True
        q_vec = q_vec.reshape(1, -1)
    
    # Compute quaternion scalar part
    qw = np.sqrt(np.clip(1.0 - np.sum(q_vec**2, axis=-1), 0.0, 1.0))
    
    # Construct full quaternion [w, x, y, z]
    quaternions = np.column_stack([qw, q_vec])
    
    # Convert to scipy format [x, y, z, w]
    quat_scipy = np.column_stack([q_vec, qw])
    
    # Convert to Euler angles (ZYX convention, which gives roll, pitch, yaw)
    rot = Rotation.from_quat(quat_scipy)
    euler = rot.as_euler('ZYX', degrees=False)  # Returns [yaw, pitch, roll]
    
    # Reorder to [roll, pitch, yaw]
    euler_reordered = np.column_stack([euler[:, 2], euler[:, 1], euler[:, 0]])
    
    if single:
        return euler_reordered[0]
    return euler_reordered


@dataclass
class SimulationResult:
    """Results from a single simulation run"""
    time: np.ndarray
    state_traj: np.ndarray  # (N, 12)
    control_traj: np.ndarray  # (N, 4)
    state_ref_traj: np.ndarray  # (N, 12)
    controller_name: str


class ComparisonSimulator:
    """
    Simulator for comparing different controllers.
    """
    
    def __init__(self, phy_params: PhyParams, dt: float = 0.01):
        """
        Initialize comparison simulator.
        
        Args:
            phy_params: Physical parameters
            dt: Simulation time step (s)
        """
        self.params = phy_params
        self.dt = dt
        self.dynamics = RocketDynamics(phy_params)
        
        # Initialize controllers
        self.lqr_full = LQRFullStateController(phy_params)
        self.lqr_att = LQRAttitudeOnlyController(phy_params)
        # Initialize PID with default limits (15 deg max deflection, 86 deg/s max angular velocity)
        self.pid = PIDController(phy_params, max_deflection_angle_deg=15.0, max_angular_velocity_deg_s=86.0)
    
    def simulate(self, controller_name: str, state0: np.ndarray, 
                 state_ref: np.ndarray, t_end: float,
                 apply_constraints: bool = True) -> SimulationResult:
        """
        Run simulation with specified controller.
        
        Args:
            controller_name: 'lqr_full', 'lqr_attitude', or 'pid'
            state0: Initial state (12D)
            state_ref: Reference state (12D) - can be constant or time-varying
            t_end: Simulation end time (s)
            apply_constraints: Whether to apply control input constraints
            
        Returns:
            SimulationResult object
        """
        # Select controller
        if controller_name == 'lqr_full':
            controller = self.lqr_full
        elif controller_name == 'lqr_attitude':
            controller = self.lqr_att
        elif controller_name == 'pid':
            controller = self.pid
            controller.reset()  # Reset PID integrators
        else:
            raise ValueError(f"Unknown controller: {controller_name}")
        
        # Initialize trajectory storage
        n_steps = int(t_end / self.dt) + 1
        time = np.linspace(0, t_end, n_steps)
        state_traj = np.zeros((n_steps, 12))
        control_traj = np.zeros((n_steps, 4))
        state_ref_traj = np.zeros((n_steps, 12))
        
        # Initial state
        state = state0.copy()
        state_traj[0] = state
        
        # Control constraints
        phi_min, phi_max = -0.2, 0.2  # rad
        theta_min, theta_max = -0.2, 0.2  # rad
        thrust_min, thrust_max = 0.0, 2.0 * self.params.MASS * self.params.G  # N
        tau_r_min, tau_r_max = -0.5, 0.5  # Nm
        
        # Simulation loop
        for i in range(1, n_steps):
            # Get reference state (can be time-varying)
            if callable(state_ref):
                state_ref_current = state_ref(time[i])
            else:
                state_ref_current = state_ref
            state_ref_traj[i] = state_ref_current
            
            # Compute control
            if controller_name == 'pid':
                u = controller.compute_control(state, state_ref_current, self.dt)
            else:
                u = controller.compute_control(state, state_ref_current)
            
            # Apply constraints
            if apply_constraints:
                u[0] = np.clip(u[0], phi_min, phi_max)
                u[1] = np.clip(u[1], theta_min, theta_max)
                u[2] = np.clip(u[2], thrust_min, thrust_max)
                u[3] = np.clip(u[3], tau_r_min, tau_r_max)
            
            control_traj[i] = u
            
            # Integrate dynamics
            state = self.dynamics.simulate_step(state, u, self.dt)
            state_traj[i] = state
        
        return SimulationResult(
            time=time,
            state_traj=state_traj,
            control_traj=control_traj,
            state_ref_traj=state_ref_traj,
            controller_name=controller_name
        )
    
    def compare_controllers(self, state0: np.ndarray, state_ref: np.ndarray,
                          t_end: float, controllers: Optional[List[str]] = None) -> Dict[str, SimulationResult]:
        """
        Compare multiple controllers on the same scenario.
        
        Args:
            state0: Initial state
            state_ref: Reference state
            t_end: Simulation end time
            controllers: List of controller names to compare. If None, compares all.
            
        Returns:
            Dictionary mapping controller names to results
        """
        if controllers is None:
            controllers = ['lqr_full', 'lqr_attitude', 'pid']
        
        results = {}
        for ctrl_name in controllers:
            print(f"Simulating with {ctrl_name}...")
            results[ctrl_name] = self.simulate(ctrl_name, state0, state_ref, t_end)
        
        return results
    
    def save_state_data(self, results: Dict[str, SimulationResult], 
                       output_dir: str = "simulation_results"):
        """
        Save state trajectories and control inputs to CSV files.
        
        Args:
            results: Dictionary of simulation results
            output_dir: Directory to save data files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, result in results.items():
            # Create filename
            filename = os.path.join(output_dir, f"{name}_state_data.csv")
            
            # Prepare data: time, state (12), control (4), reference (12)
            data = np.hstack([
                result.time.reshape(-1, 1),
                result.state_traj,
                result.control_traj,
                result.state_ref_traj
            ])
            
            # Create header
            header = "time,"
            header += "x,y,z,vx,vy,vz,qx,qy,qz,p,q,r,"
            header += "phi,theta,thrust,tau_r,"
            header += "x_ref,y_ref,z_ref,vx_ref,vy_ref,vz_ref,qx_ref,qy_ref,qz_ref,p_ref,q_ref,r_ref"
            
            # Save to CSV
            np.savetxt(filename, data, delimiter=',', header=header, comments='')
            print(f"Saved state data for {name} to {filename}")
    
    def plot_comparison(self, results: Dict[str, SimulationResult], 
                       save_path: Optional[str] = None, show_control: bool = True,
                       show_errors: bool = True, show_3d_trajectory: bool = True):
        """
        Plot comprehensive comparison of different controllers.
        
        Args:
            results: Dictionary of simulation results
            save_path: Base path to save figures (optional, will add suffixes)
            show_control: Whether to plot control inputs
            show_errors: Whether to plot tracking errors
            show_3d_trajectory: Whether to plot 3D trajectory
        """
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        linestyles = ['-', '-', '-', '-', '-', '-']
        
        # ========== State Trajectories ==========
        fig1, axes1 = plt.subplots(4, 3, figsize=(16, 12))
        fig1.suptitle('State Trajectories Comparison', fontsize=16, fontweight='bold')
        
        # Position plots
        axes1[0, 0].set_title('Position X (m)', fontsize=12)
        axes1[0, 0].set_ylabel('X (m)')
        axes1[0, 1].set_title('Position Y (m)', fontsize=12)
        axes1[0, 1].set_ylabel('Y (m)')
        axes1[0, 2].set_title('Position Z (m)', fontsize=12)
        axes1[0, 2].set_ylabel('Z (m)')
        
        # Velocity plots
        axes1[1, 0].set_title('Velocity Vx (m/s)', fontsize=12)
        axes1[1, 0].set_ylabel('Vx (m/s)')
        axes1[1, 1].set_title('Velocity Vy (m/s)', fontsize=12)
        axes1[1, 1].set_ylabel('Vy (m/s)')
        axes1[1, 2].set_title('Velocity Vz (m/s)', fontsize=12)
        axes1[1, 2].set_ylabel('Vz (m/s)')
        
        # Attitude plots (Euler angles)
        axes1[2, 0].set_title('Attitude Roll (deg)', fontsize=12)
        axes1[2, 0].set_ylabel('Roll (deg)')
        axes1[2, 1].set_title('Attitude Pitch (deg)', fontsize=12)
        axes1[2, 1].set_ylabel('Pitch (deg)')
        axes1[2, 2].set_title('Attitude Yaw (deg)', fontsize=12)
        axes1[2, 2].set_ylabel('Yaw (deg)')
        
        # Angular velocity plots
        axes1[3, 0].set_title('Angular Velocity P (deg/s)', fontsize=12)
        axes1[3, 0].set_ylabel('P (deg/s)')
        axes1[3, 1].set_title('Angular Velocity Q (deg/s)', fontsize=12)
        axes1[3, 1].set_ylabel('Q (deg/s)')
        axes1[3, 2].set_title('Angular Velocity R (deg/s)', fontsize=12)
        axes1[3, 2].set_ylabel('R (deg/s)')
        
        for idx, (name, result) in enumerate(results.items()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            time = result.time
            
            # Position
            axes1[0, 0].plot(time, result.state_traj[:, 0], color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[0, 0].plot(time, result.state_ref_traj[:, 0], color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            axes1[0, 1].plot(time, result.state_traj[:, 1], color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[0, 1].plot(time, result.state_ref_traj[:, 1], color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            axes1[0, 2].plot(time, result.state_traj[:, 2], color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[0, 2].plot(time, result.state_ref_traj[:, 2], color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            
            # Velocity
            axes1[1, 0].plot(time, result.state_traj[:, 3], color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[1, 0].plot(time, result.state_ref_traj[:, 3], color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            axes1[1, 1].plot(time, result.state_traj[:, 4], color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[1, 1].plot(time, result.state_ref_traj[:, 4], color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            axes1[1, 2].plot(time, result.state_traj[:, 5], color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[1, 2].plot(time, result.state_ref_traj[:, 5], color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            
            # Attitude (convert quaternion to Euler angles)
            euler_traj = quaternion_to_euler(result.state_traj[:, 6:9])  # [roll, pitch, yaw] in rad
            euler_ref = quaternion_to_euler(result.state_ref_traj[:, 6:9])
            
            # Convert to degrees
            euler_traj_deg = np.degrees(euler_traj)
            euler_ref_deg = np.degrees(euler_ref)
            
            axes1[2, 0].plot(time, euler_traj_deg[:, 0], color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[2, 0].plot(time, euler_ref_deg[:, 0], color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            axes1[2, 1].plot(time, euler_traj_deg[:, 1], color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[2, 1].plot(time, euler_ref_deg[:, 1], color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            axes1[2, 2].plot(time, euler_traj_deg[:, 2], color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[2, 2].plot(time, euler_ref_deg[:, 2], color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            
            # Angular velocity (convert from rad/s to deg/s)
            axes1[3, 0].plot(time, np.degrees(result.state_traj[:, 9]), color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[3, 0].plot(time, np.degrees(result.state_ref_traj[:, 9]), color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            axes1[3, 1].plot(time, np.degrees(result.state_traj[:, 10]), color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[3, 1].plot(time, np.degrees(result.state_ref_traj[:, 10]), color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
            axes1[3, 2].plot(time, np.degrees(result.state_traj[:, 11]), color=color, label=name, 
                           linestyle=linestyle, linewidth=2)
            axes1[3, 2].plot(time, np.degrees(result.state_ref_traj[:, 11]), color=color, 
                           linestyle='--', alpha=0.4, linewidth=1)
        
        # Add legends and formatting
        for ax in axes1.flat:
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig1.savefig(f"{save_path}_states.png", dpi=300, bbox_inches='tight')
            print(f"Saved state trajectories plot to {save_path}_states.png")
        
        # ========== Control Inputs ==========
        if show_control:
            fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
            fig2.suptitle('Control Inputs Comparison', fontsize=16, fontweight='bold')
            
            axes2[0, 0].set_title('Thrust Deflection Angle Phi (deg)', fontsize=12)
            axes2[0, 0].set_ylabel('Phi (deg)')
            axes2[0, 1].set_title('Thrust Deflection Angle Theta (deg)', fontsize=12)
            axes2[0, 1].set_ylabel('Theta (deg)')
            axes2[1, 0].set_title('Thrust Force (N)', fontsize=12)
            axes2[1, 0].set_ylabel('Thrust (N)')
            axes2[1, 1].set_title('Yaw Torque (Nm)', fontsize=12)
            axes2[1, 1].set_ylabel('Tau_r (Nm)')
            
            for idx, (name, result) in enumerate(results.items()):
                color = colors[idx % len(colors)]
                linestyle = linestyles[idx % len(linestyles)]
                time = result.time
                
                # Convert phi and theta from radians to degrees
                axes2[0, 0].plot(time, np.degrees(result.control_traj[:, 0]), color=color, 
                               label=name, linestyle=linestyle, linewidth=2)
                axes2[0, 1].plot(time, np.degrees(result.control_traj[:, 1]), color=color, 
                               label=name, linestyle=linestyle, linewidth=2)
                axes2[1, 0].plot(time, result.control_traj[:, 2], color=color, 
                               label=name, linestyle=linestyle, linewidth=2)
                axes2[1, 1].plot(time, result.control_traj[:, 3], color=color, 
                               label=name, linestyle=linestyle, linewidth=2)
            
            for ax in axes2.flat:
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time (s)', fontsize=10)
            
            plt.tight_layout()
            
            if save_path:
                fig2.savefig(f"{save_path}_controls.png", dpi=300, bbox_inches='tight')
                print(f"Saved control inputs plot to {save_path}_controls.png")
        
        # ========== Tracking Errors ==========
        if show_errors:
            fig3, axes3 = plt.subplots(4, 3, figsize=(16, 12))
            fig3.suptitle('Tracking Errors Comparison', fontsize=16, fontweight='bold')
            
            error_labels = [
                ['Position Error X (m)', 'Position Error Y (m)', 'Position Error Z (m)'],
                ['Velocity Error Vx (m/s)', 'Velocity Error Vy (m/s)', 'Velocity Error Vz (m/s)'],
                ['Attitude Error Roll (deg)', 'Attitude Error Pitch (deg)', 'Attitude Error Yaw (deg)'],
                ['Angular Vel Error P (deg/s)', 'Angular Vel Error Q (deg/s)', 'Angular Vel Error R (deg/s)']
            ]
            
            for idx, (name, result) in enumerate(results.items()):
                color = colors[idx % len(colors)]
                linestyle = linestyles[idx % len(linestyles)]
                time = result.time
                
                for i in range(4):
                    for j in range(3):
                        state_idx = i * 3 + j
                        if i == 2:  # Attitude row - convert to Euler angles
                            # Convert quaternion to Euler angles
                            euler_traj = quaternion_to_euler(result.state_traj[:, 6:9])
                            euler_ref = quaternion_to_euler(result.state_ref_traj[:, 6:9])
                            euler_error = euler_traj - euler_ref
                            # Convert to degrees
                            euler_error_deg = np.degrees(euler_error[:, j])
                            axes3[i, j].plot(time, euler_error_deg, color=color, 
                                           label=name, linestyle=linestyle, linewidth=2)
                        elif i == 3:  # Angular velocity row - convert from rad/s to deg/s
                            error = result.state_traj[:, state_idx] - result.state_ref_traj[:, state_idx]
                            error_deg = np.degrees(error)  # Convert from rad/s to deg/s
                            axes3[i, j].plot(time, error_deg, color=color, 
                                           label=name, linestyle=linestyle, linewidth=2)
                        else:
                            error = result.state_traj[:, state_idx] - result.state_ref_traj[:, state_idx]
                            axes3[i, j].plot(time, error, color=color, 
                                           label=name, linestyle=linestyle, linewidth=2)
                        axes3[i, j].set_title(error_labels[i][j], fontsize=12)
                        axes3[i, j].set_ylabel(error_labels[i][j].split()[0], fontsize=10)
            
            for ax in axes3.flat:
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time (s)', fontsize=10)
                ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
            
            plt.tight_layout()
            
            if save_path:
                fig3.savefig(f"{save_path}_errors.png", dpi=300, bbox_inches='tight')
                print(f"Saved tracking errors plot to {save_path}_errors.png")
        
        # ========== 3D Trajectory ==========
        if show_3d_trajectory:
            fig4 = plt.figure(figsize=(12, 10))
            ax4 = fig4.add_subplot(111, projection='3d')
            ax4.set_title('3D Trajectory Comparison', fontsize=16, fontweight='bold')
            
            for idx, (name, result) in enumerate(results.items()):
                color = colors[idx % len(colors)]
                linestyle = linestyles[idx % len(linestyles)]
                
                # Plot trajectory
                ax4.plot(result.state_traj[:, 0], result.state_traj[:, 1], 
                        result.state_traj[:, 2], color=color, label=name, 
                        linestyle=linestyle, linewidth=2, alpha=0.8)
                
                # Mark start and end points
                ax4.scatter(result.state_traj[0, 0], result.state_traj[0, 1], 
                          result.state_traj[0, 2], color=color, s=100, marker='o', 
                          label=f'{name} start')
                ax4.scatter(result.state_traj[-1, 0], result.state_traj[-1, 1], 
                          result.state_traj[-1, 2], color=color, s=100, marker='s', 
                          label=f'{name} end')
            
            ax4.set_xlabel('X (m)', fontsize=12)
            ax4.set_ylabel('Y (m)', fontsize=12)
            ax4.set_zlabel('Z (m)', fontsize=12)
            ax4.legend(loc='best', fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                fig4.savefig(f"{save_path}_3d_trajectory.png", dpi=300, bbox_inches='tight')
                print(f"Saved 3D trajectory plot to {save_path}_3d_trajectory.png")
        
        # Show all plots
        if not save_path:
            plt.show()
        else:
            plt.close('all')
    
    def compute_metrics(self, results: Dict[str, SimulationResult]) -> Dict[str, Dict[str, float]]:
        """
        Compute performance metrics for each controller.
        
        Args:
            results: Dictionary of simulation results
            
        Returns:
            Dictionary mapping controller names to metrics
        """
        metrics = {}
        
        for name, result in results.items():
            # Compute errors
            error_traj = result.state_traj - result.state_ref_traj
            
            # RMSE for each state component
            rmse = np.sqrt(np.mean(error_traj**2, axis=0))
            
            # Total RMSE
            total_rmse = np.sqrt(np.mean(error_traj**2))
            
            # Maximum error
            max_error = np.max(np.abs(error_traj), axis=0)
            
            # Control effort (integral of squared control)
            control_effort = np.sum(result.control_traj**2, axis=0)
            total_control_effort = np.sum(result.control_traj**2)
            
            metrics[name] = {
                'rmse_total': total_rmse,
                'rmse_pos': np.linalg.norm(rmse[0:3]),
                'rmse_vel': np.linalg.norm(rmse[3:6]),
                'rmse_att': np.linalg.norm(rmse[6:9]),
                'rmse_omega': np.linalg.norm(rmse[9:12]),
                'max_error_total': np.max(max_error),
                'control_effort_total': total_control_effort,
                'control_effort_phi': control_effort[0],
                'control_effort_theta': control_effort[1],
                'control_effort_thrust': control_effort[2],
                'control_effort_tau_r': control_effort[3],
            }
        
        return metrics


def main():
    """Example usage of the comparison simulator."""
    # Physical parameters (from tvc_params.yaml)
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
    
    # Initial state (slightly perturbed from reference)
    state0 = np.array([
        0.0, 0.0, 0.0,  # position
        0.0, 0.0, 0.0,  # velocity
        0.0, 0.0, 0.0,  # attitude
        0.0, 0.0, 0.0   # angular velocity
    ])
    
    # Reference state (hover at origin)
    state_ref = np.array([
        0.0, 0.0, 1.0,  # position (1m above ground in ENU)
        0.0, 0.0, 0.0,  # velocity
        0.0, 0.0, 0.0,  # attitude
        0.0, 0.0, 0.0   # angular velocity
    ])
    
    # Run comparison
    print("Running controller comparison simulations...")
    results = simulator.compare_controllers(state0, state_ref, t_end=5.0)
    
    # Save state data to CSV files
    print("\nSaving state data to CSV files...")
    simulator.save_state_data(results, output_dir="simulation_results")
    
    # Plot comprehensive comparison
    print("\nGenerating plots...")
    simulator.plot_comparison(
        results, 
        save_path='controller_comparison',
        show_control=True,
        show_errors=True,
        show_3d_trajectory=True
    )
    
    # Print metrics
    metrics = simulator.compute_metrics(results)
    print("\n" + "=" * 80)
    print("Performance Metrics:")
    print("=" * 80)
    for name, m in metrics.items():
        print(f"\n{name}:")
        print(f"  Total RMSE: {m['rmse_total']:.4f}")
        print(f"  Position RMSE: {m['rmse_pos']:.4f}")
        print(f"  Velocity RMSE: {m['rmse_vel']:.4f}")
        print(f"  Attitude RMSE: {m['rmse_att']:.4f}")
        print(f"  Angular Velocity RMSE: {m['rmse_omega']:.4f}")
        print(f"  Total Control Effort: {m['control_effort_total']:.4f}")
    print("\n" + "=" * 80)
    print("All plots saved. Check the following files:")
    print("  - controller_comparison_states.png")
    print("  - controller_comparison_controls.png")
    print("  - controller_comparison_errors.png")
    print("  - controller_comparison_3d_trajectory.png")
    print("  - simulation_results/*.csv (state data files)")
    print("=" * 80)


if __name__ == "__main__":
    main()
