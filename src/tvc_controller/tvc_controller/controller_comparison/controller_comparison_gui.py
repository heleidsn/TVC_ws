#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5 GUI for Controller Comparison

Provides a graphical interface for comparing different controllers:
- Full-state LQR
- Attitude-only LQR
- PID
"""

import sys
import os
import numpy as np

# Set environment variables for high DPI scaling before importing Qt
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'
os.environ['QT_SCALE_FACTOR'] = '1'

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QCoreApplication
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QCheckBox, QLabel, QGroupBox, QTextEdit,
                             QProgressBar, QMessageBox, QTabWidget, QDoubleSpinBox,
                             QFormLayout, QScrollArea, QTableWidget, QTableWidgetItem, QFrame)
from PyQt5.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 100
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# Handle imports
if __name__ == "__main__" or __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from controller_comparison.rocket_dynamics import PhyParams, RocketDynamics
    from controller_comparison.lqr_full_state import LQRFullStateController
    from controller_comparison.lqr_attitude_only import LQRAttitudeOnlyController
    from controller_comparison.pid_controller import PIDController
    from controller_comparison.comparison_simulator import ComparisonSimulator, SimulationResult
else:
    from .rocket_dynamics import PhyParams, RocketDynamics
    from .lqr_full_state import LQRFullStateController
    from .lqr_attitude_only import LQRAttitudeOnlyController
    from .pid_controller import PIDController
    from .comparison_simulator import ComparisonSimulator, SimulationResult


class SimulationWorker(QThread):
    """Worker thread for running simulations without blocking the GUI"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, simulator, controllers, state0, state_ref, t_end):
        super().__init__()
        self.simulator = simulator
        self.controllers = controllers
        self.state0 = state0
        self.state_ref = state_ref
        self.t_end = t_end
        
    def run(self):
        try:
            results = {}
            for i, ctrl_name in enumerate(self.controllers):
                self.progress.emit(f"Simulating with {ctrl_name}... ({i+1}/{len(self.controllers)})")
                result = self.simulator.simulate(ctrl_name, self.state0, self.state_ref, self.t_end)
                results[ctrl_name] = result
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class AttitudeDebugWorker(QThread):
    """Worker thread for running attitude control debug simulation"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, phy_params, att_init_euler, att_ref_euler, sim_time, dt):
        super().__init__()
        self.phy_params = phy_params
        self.att_init_euler = att_init_euler
        self.att_ref_euler = att_ref_euler
        self.sim_time = sim_time
        self.dt = dt
        
    def run(self):
        try:
            # Use already imported modules
            if __name__ == "__main__" or __package__ is None:
                from controller_comparison.pid_controller import PIDController
                from controller_comparison.rocket_dynamics import RocketDynamics
            else:
                from .pid_controller import PIDController
                from .rocket_dynamics import RocketDynamics
            
            # Initialize controller and dynamics
            pid = PIDController(self.phy_params)
            dynamics = RocketDynamics(self.phy_params)
            
            # Convert initial attitude to quaternion
            rot_init = Rotation.from_euler('ZYX', 
                [self.att_init_euler[2], self.att_init_euler[1], self.att_init_euler[0]], 
                degrees=False)
            quat_init = rot_init.as_quat()  # [x, y, z, w]
            att_init = quat_init[:3]  # qx, qy, qz
            
            # Initial state: position at 1m altitude, zero velocity, initial attitude, zero angular velocity
            state0 = np.array([
                0.0, 0.0, 1.0,  # Position: x, y, z (1m altitude)
                0.0, 0.0, 0.0,  # Velocity: vx, vy, vz
                att_init[0], att_init[1], att_init[2],  # Attitude: qx, qy, qz
                0.0, 0.0, 0.0   # Angular velocity: p, q, r
            ])
            
            # Storage for results
            time_points = []
            att_euler_traj = []
            att_ref_euler_traj = []
            omega_traj = []
            control_traj = []
            
            # Reset controller
            pid.reset()
            
            # Simulation loop
            t = 0.0
            state = state0.copy()
            num_steps = int(self.sim_time / self.dt)
            
            self.progress.emit("Running attitude control simulation...")
            
            for step in range(num_steps):
                t = step * self.dt
                
                # Compute control using attitude-only mode
                u = pid.compute_attitude_control_only(state, self.att_ref_euler, self.dt)
                
                # Compute dynamics
                state_dot = dynamics.compute_dynamics(t, state, u)
                
                # Integrate
                state = state + state_dot * self.dt
                
                # Convert to Euler angles
                q_vec = state[6:9]
                qw = np.sqrt(np.clip(1.0 - np.sum(q_vec**2), 0.0, 1.0))
                quat = np.array([qw, q_vec[0], q_vec[1], q_vec[2]])
                rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                euler = rot.as_euler('ZYX', degrees=False)  # [yaw, pitch, roll]
                att_euler = np.array([euler[2], euler[1], euler[0]])  # [roll, pitch, yaw]
                
                # Store for plotting
                time_points.append(t)
                att_euler_traj.append(np.rad2deg(att_euler))
                att_ref_euler_traj.append(np.rad2deg(self.att_ref_euler))
                omega_traj.append(np.rad2deg(state[9:12]))
                control_traj.append(u.copy())
                
                if step % 100 == 0:
                    self.progress.emit(f"Step {step}/{num_steps}...")
            
            # Convert to numpy arrays
            result = {
                'time': np.array(time_points),
                'att_euler': np.array(att_euler_traj),
                'att_ref_euler': np.array(att_ref_euler_traj),
                'omega': np.array(omega_traj),
                'control': np.array(control_traj),
                'att_ref_euler_deg': np.rad2deg(self.att_ref_euler)
            }
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MatplotlibWidget(QWidget):
    """Widget for displaying matplotlib figures"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def clear(self):
        """Clear the figure"""
        self.figure.clear()
        self.canvas.draw()
    
    def plot_states(self, results):
        """Plot state trajectories"""
        self.figure.clear()
        self.figure.set_size_inches(12, 10)
        axes = self.figure.subplots(4, 3)
        self.figure.suptitle('State Trajectories Comparison', fontsize=14, fontweight='bold')
        
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        linestyles = ['-', '-', '-', '-', '-', '-']
        
        state_labels = [
            ['Position X (m)', 'Position Y (m)', 'Position Z (m)'],
            ['Velocity Vx (m/s)', 'Velocity Vy (m/s)', 'Velocity Vz (m/s)'],
            ['Attitude Roll (deg)', 'Attitude Pitch (deg)', 'Attitude Yaw (deg)'],
            ['Angular Vel P (deg/s)', 'Angular Vel Q (deg/s)', 'Angular Vel R (deg/s)']
        ]
        
        def quaternion_to_euler(q_vec):
            """Convert quaternion to Euler angles"""
            single = False
            if q_vec.ndim == 1:
                single = True
                q_vec = q_vec.reshape(1, -1)
            qw = np.sqrt(np.clip(1.0 - np.sum(q_vec**2, axis=-1), 0.0, 1.0))
            quat_scipy = np.column_stack([q_vec, qw])
            rot = Rotation.from_quat(quat_scipy)
            euler = rot.as_euler('ZYX', degrees=False)
            euler_reordered = np.column_stack([euler[:, 2], euler[:, 1], euler[:, 0]])
            if single:
                return euler_reordered[0]
            return euler_reordered
        
        for idx, (name, result) in enumerate(results.items()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            time = result.time
            
            for i in range(4):
                for j in range(3):
                    state_idx = i * 3 + j
                    if i == 2:  # Attitude row - convert to Euler angles
                        euler_traj = quaternion_to_euler(result.state_traj[:, 6:9])
                        euler_ref = quaternion_to_euler(result.state_ref_traj[:, 6:9])
                        axes[i, j].plot(time, np.degrees(euler_traj[:, j]), 
                                       color=color, label=name, linestyle=linestyle, linewidth=2)
                        axes[i, j].plot(time, np.degrees(euler_ref[:, j]), 
                                       color=color, linestyle='--', alpha=0.4, linewidth=1)
                    elif i == 3:  # Angular velocity row - convert from rad/s to deg/s
                        axes[i, j].plot(time, np.degrees(result.state_traj[:, state_idx]), 
                                       color=color, label=name, linestyle=linestyle, linewidth=2)
                        axes[i, j].plot(time, np.degrees(result.state_ref_traj[:, state_idx]), 
                                       color=color, linestyle='--', alpha=0.4, linewidth=1)
                    else:
                        axes[i, j].plot(time, result.state_traj[:, state_idx], 
                                       color=color, label=name, linestyle=linestyle, linewidth=2)
                        axes[i, j].plot(time, result.state_ref_traj[:, state_idx], 
                                       color=color, linestyle='--', alpha=0.4, linewidth=1)
                    axes[i, j].set_title(state_labels[i][j], fontsize=10)
                    axes[i, j].grid(True, alpha=0.3)
                    axes[i, j].set_xlabel('Time (s)', fontsize=9)
        
        for ax in axes.flat:
            ax.legend(loc='best', fontsize=8)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_controls(self, results):
        """Plot control inputs"""
        self.figure.clear()
        self.figure.set_size_inches(10, 8)
        axes = self.figure.subplots(2, 2)
        self.figure.suptitle('Control Inputs Comparison', fontsize=14, fontweight='bold')
        
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        linestyles = ['-', '-', '-', '-', '-', '-']
        
        control_labels = [
            ['Thrust Deflection Phi (deg)', 'Thrust Deflection Theta (deg)'],
            ['Thrust Force (N)', 'Yaw Torque (Nm)']
        ]
        
        for idx, (name, result) in enumerate(results.items()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            time = result.time
            
            # Convert phi and theta from radians to degrees
            axes[0, 0].plot(time, np.degrees(result.control_traj[:, 0]), color=color, 
                          label=name, linestyle=linestyle, linewidth=2)
            axes[0, 1].plot(time, np.degrees(result.control_traj[:, 1]), color=color, 
                          label=name, linestyle=linestyle, linewidth=2)
            axes[1, 0].plot(time, result.control_traj[:, 2], color=color, 
                          label=name, linestyle=linestyle, linewidth=2)
            axes[1, 1].plot(time, result.control_traj[:, 3], color=color, 
                          label=name, linestyle=linestyle, linewidth=2)
        
        axes[0, 0].set_title(control_labels[0][0], fontsize=10)
        axes[0, 1].set_title(control_labels[0][1], fontsize=10)
        axes[1, 0].set_title(control_labels[1][0], fontsize=10)
        axes[1, 1].set_title(control_labels[1][1], fontsize=10)
        
        for ax in axes.flat:
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)', fontsize=9)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_errors(self, results):
        """Plot tracking errors"""
        self.figure.clear()
        self.figure.set_size_inches(12, 10)
        axes = self.figure.subplots(4, 3)
        self.figure.suptitle('Tracking Errors Comparison', fontsize=14, fontweight='bold')
        
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        linestyles = ['-', '-', '-', '-', '-', '-']
        
        error_labels = [
            ['Position Error X (m)', 'Position Error Y (m)', 'Position Error Z (m)'],
            ['Velocity Error Vx (m/s)', 'Velocity Error Vy (m/s)', 'Velocity Error Vz (m/s)'],
            ['Attitude Error Roll (deg)', 'Attitude Error Pitch (deg)', 'Attitude Error Yaw (deg)'],
            ['Angular Vel Error P (deg/s)', 'Angular Vel Error Q (deg/s)', 'Angular Vel Error R (deg/s)']
        ]
        
        def quaternion_to_euler(q_vec):
            """Convert quaternion to Euler angles"""
            single = False
            if q_vec.ndim == 1:
                single = True
                q_vec = q_vec.reshape(1, -1)
            qw = np.sqrt(np.clip(1.0 - np.sum(q_vec**2, axis=-1), 0.0, 1.0))
            quat_scipy = np.column_stack([q_vec, qw])
            rot = Rotation.from_quat(quat_scipy)
            euler = rot.as_euler('ZYX', degrees=False)
            euler_reordered = np.column_stack([euler[:, 2], euler[:, 1], euler[:, 0]])
            if single:
                return euler_reordered[0]
            return euler_reordered
        
        for idx, (name, result) in enumerate(results.items()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            time = result.time
            
            for i in range(4):
                for j in range(3):
                    state_idx = i * 3 + j
                    if i == 2:  # Attitude row - convert to Euler angles
                        euler_traj = quaternion_to_euler(result.state_traj[:, 6:9])
                        euler_ref = quaternion_to_euler(result.state_ref_traj[:, 6:9])
                        euler_error = euler_traj - euler_ref
                        axes[i, j].plot(time, np.degrees(euler_error[:, j]), color=color, 
                                       label=name, linestyle=linestyle, linewidth=2)
                    elif i == 3:  # Angular velocity row - convert from rad/s to deg/s
                        error = result.state_traj[:, state_idx] - result.state_ref_traj[:, state_idx]
                        error_deg = np.degrees(error)  # Convert from rad/s to deg/s
                        axes[i, j].plot(time, error_deg, color=color, 
                                       label=name, linestyle=linestyle, linewidth=2)
                    else:
                        error = result.state_traj[:, state_idx] - result.state_ref_traj[:, state_idx]
                        axes[i, j].plot(time, error, color=color, 
                                       label=name, linestyle=linestyle, linewidth=2)
                    axes[i, j].set_title(error_labels[i][j], fontsize=10)
                    axes[i, j].grid(True, alpha=0.3)
                    axes[i, j].set_xlabel('Time (s)', fontsize=9)
                    axes[i, j].axhline(y=0, color='k', linestyle=':', alpha=0.5)
        
        for ax in axes.flat:
            ax.legend(loc='best', fontsize=8)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_3d_trajectory(self, results):
        """Plot 3D trajectory"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        ax.set_title('3D Trajectory Comparison', fontsize=14, fontweight='bold')
        
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        linestyles = ['-', '-', '-', '-', '-', '-']
        
        for idx, (name, result) in enumerate(results.items()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            
            ax.plot(result.state_traj[:, 0], result.state_traj[:, 1], 
                   result.state_traj[:, 2], color=color, label=name, 
                   linestyle=linestyle, linewidth=2, alpha=0.8)
            
            ax.scatter(result.state_traj[0, 0], result.state_traj[0, 1], 
                      result.state_traj[0, 2], color=color, s=100, marker='o')
            ax.scatter(result.state_traj[-1, 0], result.state_traj[-1, 1], 
                      result.state_traj[-1, 2], color=color, s=100, marker='s')
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_attitude_debug(self, result):
        """Plot attitude control debug results"""
        self.figure.clear()
        self.figure.set_size_inches(16, 12)
        axes = self.figure.subplots(3, 3)
        self.figure.suptitle(f'PID Attitude Control Response\nReference: Roll={result["att_ref_euler_deg"][0]:.1f}°, '
                             f'Pitch={result["att_ref_euler_deg"][1]:.1f}°, Yaw={result["att_ref_euler_deg"][2]:.1f}°', 
                             fontsize=14)
        
        time_points = result['time']
        att_euler_traj = result['att_euler']
        att_ref_euler_traj = result['att_ref_euler']
        omega_traj = result['omega']
        control_traj = result['control']
        
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
        axes[2, 2].axhline(y=0.6570 * 9.81, color='r', linestyle='--', alpha=0.7, label='Equilibrium')
        axes[2, 2].set_xlabel('Time (s)')
        axes[2, 2].set_ylabel('Thrust (N)')
        axes[2, 2].set_title('Control Input: Thrust')
        axes[2, 2].grid(True)
        axes[2, 2].legend()
        
        self.figure.tight_layout()
        self.canvas.draw()


class ControllerComparisonGUI(QMainWindow):
    """Main GUI window for controller comparison"""
    
    def __init__(self):
        super().__init__()
        self.simulator = None
        self.results = {}
        self.worker = None
        self.att_debug_worker = None
        
        self.init_ui()
        self.init_simulator()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Controller Comparison - LQR vs PID")
        
        # Set window size
        app = QApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
            screen_geometry = screen.availableGeometry()
            window_width = int(screen_geometry.width() * 0.95)
            window_height = int(screen_geometry.height() * 0.95)
            x_pos = (screen_geometry.width() - window_width) // 2
            y_pos = (screen_geometry.height() - window_height) // 2
            self.setGeometry(x_pos, y_pos, window_width, window_height)
        else:
            self.setGeometry(50, 50, 1900, 1000)
        
        self.setMinimumSize(1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        central_widget.setLayout(main_layout)
        
        # Left panel for controls (smaller stretch factor to make it smaller)
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)  # Stretch factor: 1
        
        # Right panel for results (larger stretch factor to make it larger)
        right_panel = self.create_result_panel()
        main_layout.addWidget(right_panel, 3)  # Stretch factor: 3 (changed from 2)
    
    def create_control_panel(self):
        """Create the control panel"""
        panel = QWidget()
        # Set maximum width to make the panel smaller
        panel.setMaximumWidth(450)  # Limit maximum width
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        panel.setLayout(layout)
        
        # Title
        title = QLabel("Controller Selection")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Controller selection group (compact, one line)
        controller_group = QGroupBox("Select Controllers")
        controller_layout = QHBoxLayout()
        controller_layout.setContentsMargins(10, 5, 10, 5)
        
        self.checkbox_pid = QCheckBox("PID Controller")
        self.checkbox_pid.setChecked(True)
        controller_layout.addWidget(self.checkbox_pid)
        
        self.checkbox_lqr_full = QCheckBox("Full-State LQR")
        self.checkbox_lqr_full.setChecked(False)
        controller_layout.addWidget(self.checkbox_lqr_full)
        
        self.checkbox_lqr_att = QCheckBox("Attitude-Only LQR")
        self.checkbox_lqr_att.setChecked(False)
        controller_layout.addWidget(self.checkbox_lqr_att)
        
        controller_layout.addStretch()
        controller_group.setLayout(controller_layout)
        layout.addWidget(controller_group)
        
        # Controller parameters (tab widget)
        params_label = QLabel("Controller Parameters:")
        params_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(params_label)
        
        self.controller_params_tabs = QTabWidget()
        self.controller_params_tabs.setMaximumHeight(450)
        
        # LQR Full-State parameters tab
        lqr_full_tab = QWidget()
        lqr_full_tab_layout = QVBoxLayout()
        lqr_full_tab_layout.setContentsMargins(10, 10, 10, 10)
        lqr_full_tab.setLayout(lqr_full_tab_layout)
        
        lqr_full_scroll = QScrollArea()
        lqr_full_scroll.setWidgetResizable(True)
        lqr_full_scroll.setFrameShape(QFrame.NoFrame)
        
        lqr_full_widget = QWidget()
        lqr_full_layout = QVBoxLayout()
        lqr_full_layout.setSpacing(5)
        lqr_full_layout.setContentsMargins(5, 5, 5, 5)
        
        # Q matrix diagonal elements (12 states) - compact layout
        q_label = QLabel("<b>Q Matrix (State Weights):</b>")
        lqr_full_layout.addWidget(q_label)
        self.lqr_full_Q = {}
        state_names = ['Pos X', 'Pos Y', 'Pos Z', 'Vel Vx', 'Vel Vy', 'Vel Vz',
                      'Att Qx', 'Att Qy', 'Att Qz', 'Ang P', 'Ang Q', 'Ang R']
        default_Q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.01]
        
        # Group states in rows of 3
        for row in range(0, 12, 3):
            row_layout = QHBoxLayout()
            for col in range(3):
                if row + col < 12:
                    i = row + col
                    label = QLabel(f"{state_names[i]}:")
                    label.setMinimumWidth(60)
                    spin = QDoubleSpinBox()
                    spin.setRange(0.0, 1000.0)
                    spin.setValue(default_Q[i])
                    spin.setSingleStep(0.1)
                    spin.setDecimals(1)
                    spin.setMaximumWidth(90)
                    self.lqr_full_Q[i] = spin
                    row_layout.addWidget(label)
                    row_layout.addWidget(spin)
            row_layout.addStretch()
            lqr_full_layout.addLayout(row_layout)
        
        # R matrix diagonal elements (4 controls) - one line
        r_label = QLabel("<b>R Matrix (Control Weights):</b>")
        lqr_full_layout.addWidget(r_label)
        r_layout = QHBoxLayout()
        self.lqr_full_R = {}
        control_names = ['Phi', 'Theta', 'Thrust', 'Tau_r']
        default_R = [10.0, 10.0, 1.0, 10.0]
        for i, name in enumerate(control_names):
            label = QLabel(f"{name}:")
            label.setMinimumWidth(50)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1000.0)
            spin.setValue(default_R[i])
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setMaximumWidth(90)
            self.lqr_full_R[i] = spin
            r_layout.addWidget(label)
            r_layout.addWidget(spin)
        r_layout.addStretch()
        lqr_full_layout.addLayout(r_layout)
        
        lqr_full_widget.setLayout(lqr_full_layout)
        lqr_full_scroll.setWidget(lqr_full_widget)
        lqr_full_tab_layout.addWidget(lqr_full_scroll)
        
        # PID parameters tab (first)
        pid_tab = QWidget()
        pid_tab_layout = QVBoxLayout()
        pid_tab_layout.setContentsMargins(10, 10, 10, 10)
        pid_tab.setLayout(pid_tab_layout)
        
        pid_scroll = QScrollArea()
        pid_scroll.setWidgetResizable(True)
        pid_scroll.setFrameShape(QFrame.NoFrame)
        
        pid_widget = QWidget()
        pid_layout = QVBoxLayout()
        pid_layout.setSpacing(5)
        pid_layout.setContentsMargins(5, 5, 5, 5)
        
        # Position PID gains - compact layout (Kp, Ki, Kd in one row per axis)
        pos_label = QLabel("<b>Position PID Gains:</b>")
        pid_layout.addWidget(pos_label)
        self.pid_Kp_pos = {}
        self.pid_Ki_pos = {}
        self.pid_Kd_pos = {}
        pos_names = ['X', 'Y', 'Z']
        default_Kp_pos = [1.0, 1.0, 10.0]
        default_Ki_pos = [0.0, 0.0, 0.0]
        default_Kd_pos = [0.5, 0.5, 0.5]
        for i, name in enumerate(pos_names):
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(f"{name}:"))
            row_layout.addWidget(QLabel("Kp:"))
            spin_kp = QDoubleSpinBox()
            spin_kp.setRange(0.0, 100.0)
            spin_kp.setValue(default_Kp_pos[i])
            spin_kp.setSingleStep(0.1)
            spin_kp.setDecimals(1)
            spin_kp.setMaximumWidth(80)
            self.pid_Kp_pos[i] = spin_kp
            row_layout.addWidget(spin_kp)
            
            row_layout.addWidget(QLabel("Ki:"))
            spin_ki = QDoubleSpinBox()
            spin_ki.setRange(0.0, 10.0)
            spin_ki.setValue(default_Ki_pos[i])
            spin_ki.setSingleStep(0.01)
            spin_ki.setDecimals(1)
            spin_ki.setMaximumWidth(80)
            self.pid_Ki_pos[i] = spin_ki
            row_layout.addWidget(spin_ki)
            
            row_layout.addWidget(QLabel("Kd:"))
            spin_kd = QDoubleSpinBox()
            spin_kd.setRange(0.0, 10.0)
            spin_kd.setValue(default_Kd_pos[i])
            spin_kd.setSingleStep(0.1)
            spin_kd.setDecimals(1)
            spin_kd.setMaximumWidth(80)
            self.pid_Kd_pos[i] = spin_kd
            row_layout.addWidget(spin_kd)
            row_layout.addStretch()
            pid_layout.addLayout(row_layout)
        
        # Velocity PID gains - compact layout
        vel_label = QLabel("<b>Velocity PID Gains:</b>")
        pid_layout.addWidget(vel_label)
        self.pid_Kp_vel = {}
        self.pid_Ki_vel = {}
        self.pid_Kd_vel = {}
        vel_names = ['Vx', 'Vy', 'Vz']
        default_Kp_vel = [1.0, 1.0, 1.0]
        default_Ki_vel = [0.0, 0.0, 0.0]
        default_Kd_vel = [0.1, 0.1, 0.1]
        for i, name in enumerate(vel_names):
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(f"{name}:"))
            row_layout.addWidget(QLabel("Kp:"))
            spin_kp = QDoubleSpinBox()
            spin_kp.setRange(0.0, 100.0)
            spin_kp.setValue(default_Kp_vel[i])
            spin_kp.setSingleStep(0.1)
            spin_kp.setDecimals(1)
            spin_kp.setMaximumWidth(80)
            self.pid_Kp_vel[i] = spin_kp
            row_layout.addWidget(spin_kp)
            
            row_layout.addWidget(QLabel("Ki:"))
            spin_ki = QDoubleSpinBox()
            spin_ki.setRange(0.0, 10.0)
            spin_ki.setValue(default_Ki_vel[i])
            spin_ki.setSingleStep(0.01)
            spin_ki.setDecimals(1)
            spin_ki.setMaximumWidth(80)
            self.pid_Ki_vel[i] = spin_ki
            row_layout.addWidget(spin_ki)
            
            row_layout.addWidget(QLabel("Kd:"))
            spin_kd = QDoubleSpinBox()
            spin_kd.setRange(0.0, 10.0)
            spin_kd.setValue(default_Kd_vel[i])
            spin_kd.setSingleStep(0.1)
            spin_kd.setDecimals(1)
            spin_kd.setMaximumWidth(80)
            self.pid_Kd_vel[i] = spin_kd
            row_layout.addWidget(spin_kd)
            row_layout.addStretch()
            pid_layout.addLayout(row_layout)
        
        # Attitude PID gains - compact layout
        att_label = QLabel("<b>Attitude PID Gains:</b>")
        pid_layout.addWidget(att_label)
        self.pid_Kp_att = {}
        self.pid_Ki_att = {}
        self.pid_Kd_att = {}
        att_names = ['Qx', 'Qy', 'Qz']
        default_Kp_att = [2.0, 2.0, 0.5]
        default_Ki_att = [0.0, 0.0, 0.0]
        default_Kd_att = [0.5, 0.5, 0.1]
        for i, name in enumerate(att_names):
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(f"{name}:"))
            row_layout.addWidget(QLabel("Kp:"))
            spin_kp = QDoubleSpinBox()
            spin_kp.setRange(0.0, 100.0)
            spin_kp.setValue(default_Kp_att[i])
            spin_kp.setSingleStep(0.1)
            spin_kp.setDecimals(1)
            spin_kp.setMaximumWidth(80)
            self.pid_Kp_att[i] = spin_kp
            row_layout.addWidget(spin_kp)
            
            row_layout.addWidget(QLabel("Ki:"))
            spin_ki = QDoubleSpinBox()
            spin_ki.setRange(0.0, 10.0)
            spin_ki.setValue(default_Ki_att[i])
            spin_ki.setSingleStep(0.01)
            spin_ki.setDecimals(1)
            spin_ki.setMaximumWidth(80)
            self.pid_Ki_att[i] = spin_ki
            row_layout.addWidget(spin_ki)
            
            row_layout.addWidget(QLabel("Kd:"))
            spin_kd = QDoubleSpinBox()
            spin_kd.setRange(0.0, 10.0)
            spin_kd.setValue(default_Kd_att[i])
            spin_kd.setSingleStep(0.1)
            spin_kd.setDecimals(1)
            spin_kd.setMaximumWidth(80)
            self.pid_Kd_att[i] = spin_kd
            row_layout.addWidget(spin_kd)
            row_layout.addStretch()
            pid_layout.addLayout(row_layout)
        
        # Angular velocity PID gains - compact layout
        omega_label = QLabel("<b>Angular Velocity PID Gains:</b>")
        pid_layout.addWidget(omega_label)
        self.pid_Kp_omega = {}
        self.pid_Ki_omega = {}
        self.pid_Kd_omega = {}
        omega_names = ['P', 'Q', 'R']
        default_Kp_omega = [1.0, 1.0, 0.5]
        default_Ki_omega = [0.0, 0.0, 0.0]
        default_Kd_omega = [0.1, 0.1, 0.05]
        for i, name in enumerate(omega_names):
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(f"{name}:"))
            row_layout.addWidget(QLabel("Kp:"))
            spin_kp = QDoubleSpinBox()
            spin_kp.setRange(0.0, 100.0)
            spin_kp.setValue(default_Kp_omega[i])
            spin_kp.setSingleStep(0.1)
            spin_kp.setDecimals(1)
            spin_kp.setMaximumWidth(80)
            self.pid_Kp_omega[i] = spin_kp
            row_layout.addWidget(spin_kp)
            
            row_layout.addWidget(QLabel("Ki:"))
            spin_ki = QDoubleSpinBox()
            spin_ki.setRange(0.0, 10.0)
            spin_ki.setValue(default_Ki_omega[i])
            spin_ki.setSingleStep(0.01)
            spin_ki.setDecimals(1)
            spin_ki.setMaximumWidth(80)
            self.pid_Ki_omega[i] = spin_ki
            row_layout.addWidget(spin_ki)
            
            row_layout.addWidget(QLabel("Kd:"))
            spin_kd = QDoubleSpinBox()
            spin_kd.setRange(0.0, 10.0)
            spin_kd.setValue(default_Kd_omega[i])
            spin_kd.setSingleStep(0.1)
            spin_kd.setDecimals(1)
            spin_kd.setMaximumWidth(80)
            self.pid_Kd_omega[i] = spin_kd
            row_layout.addWidget(spin_kd)
            row_layout.addStretch()
            pid_layout.addLayout(row_layout)
        
        pid_widget.setLayout(pid_layout)
        pid_scroll.setWidget(pid_widget)
        pid_tab_layout.addWidget(pid_scroll)
        self.controller_params_tabs.addTab(pid_tab, "PID Controller")
        
        # Full-State LQR parameters tab (second)
        self.controller_params_tabs.addTab(lqr_full_tab, "Full-State LQR")
        
        # LQR Attitude-Only parameters tab (third)
        lqr_att_tab = QWidget()
        lqr_att_tab_layout = QVBoxLayout()
        lqr_att_tab_layout.setContentsMargins(10, 10, 10, 10)
        lqr_att_tab.setLayout(lqr_att_tab_layout)
        
        lqr_att_scroll = QScrollArea()
        lqr_att_scroll.setWidgetResizable(True)
        lqr_att_scroll.setFrameShape(QFrame.NoFrame)
        
        lqr_att_widget = QWidget()
        lqr_att_layout = QVBoxLayout()
        lqr_att_layout.setSpacing(5)
        lqr_att_layout.setContentsMargins(5, 5, 5, 5)
        
        # Q_att matrix diagonal elements (6 states) - compact layout
        q_label = QLabel("<b>Q Matrix (State Weights):</b>")
        lqr_att_layout.addWidget(q_label)
        q_layout = QHBoxLayout()
        self.lqr_att_Q = {}
        att_state_names = ['Att Qx', 'Att Qy', 'Att Qz', 'Ang P', 'Ang Q', 'Ang R']
        default_Q_att = [1.0, 1.0, 0.1, 1.0, 1.0, 0.01]
        for i, name in enumerate(att_state_names):
            label = QLabel(f"{name}:")
            label.setMinimumWidth(60)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1000.0)
            spin.setValue(default_Q_att[i])
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setMaximumWidth(90)
            self.lqr_att_Q[i] = spin
            q_layout.addWidget(label)
            q_layout.addWidget(spin)
        q_layout.addStretch()
        lqr_att_layout.addLayout(q_layout)
        
        # R_att matrix diagonal elements (3 controls) - one line
        r_label = QLabel("<b>R Matrix (Control Weights):</b>")
        lqr_att_layout.addWidget(r_label)
        r_layout = QHBoxLayout()
        self.lqr_att_R = {}
        att_control_names = ['Phi', 'Theta', 'Tau_r']
        default_R_att = [10.0, 10.0, 10.0]
        for i, name in enumerate(att_control_names):
            label = QLabel(f"{name}:")
            label.setMinimumWidth(50)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1000.0)
            spin.setValue(default_R_att[i])
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setMaximumWidth(90)
            self.lqr_att_R[i] = spin
            r_layout.addWidget(label)
            r_layout.addWidget(spin)
        r_layout.addStretch()
        lqr_att_layout.addLayout(r_layout)
        
        lqr_att_widget.setLayout(lqr_att_layout)
        lqr_att_scroll.setWidget(lqr_att_widget)
        lqr_att_tab_layout.addWidget(lqr_att_scroll)
        self.controller_params_tabs.addTab(lqr_att_tab, "Attitude-Only LQR")
        
        layout.addWidget(self.controller_params_tabs)
        
        # Simulation parameters group
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QFormLayout()
        
        self.spin_t_end = QDoubleSpinBox()
        self.spin_t_end.setRange(0.1, 100.0)
        self.spin_t_end.setValue(10.0)
        self.spin_t_end.setSingleStep(0.5)
        self.spin_t_end.setSuffix(" s")
        params_layout.addRow("Simulation Time:", self.spin_t_end)
        
        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setRange(0.001, 0.1)
        self.spin_dt.setValue(0.01)
        self.spin_dt.setSingleStep(0.001)
        self.spin_dt.setDecimals(3)
        self.spin_dt.setSuffix(" s")
        params_layout.addRow("Time Step:", self.spin_dt)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Initial state group (compact, one line)
        init_state_group = QGroupBox("Initial State")
        init_layout = QHBoxLayout()
        init_layout.setContentsMargins(10, 5, 10, 5)
        
        init_layout.addWidget(QLabel("X (m):"))
        self.spin_x0 = QDoubleSpinBox()
        self.spin_x0.setRange(-10.0, 10.0)
        self.spin_x0.setValue(0.0)
        self.spin_x0.setDecimals(1)
        self.spin_x0.setMaximumWidth(100)
        init_layout.addWidget(self.spin_x0)
        
        init_layout.addWidget(QLabel("Y (m):"))
        self.spin_y0 = QDoubleSpinBox()
        self.spin_y0.setRange(-10.0, 10.0)
        self.spin_y0.setValue(0.0)
        self.spin_y0.setDecimals(1)
        self.spin_y0.setMaximumWidth(100)
        init_layout.addWidget(self.spin_y0)
        
        init_layout.addWidget(QLabel("Z (m):"))
        self.spin_z0 = QDoubleSpinBox()
        self.spin_z0.setRange(0.0, 10.0)  # ENU: z >= 0 (above ground)
        self.spin_z0.setValue(0.0)  # Ground level in ENU
        self.spin_z0.setDecimals(1)
        self.spin_z0.setMaximumWidth(100)
        init_layout.addWidget(self.spin_z0)
        
        init_layout.addStretch()
        init_state_group.setLayout(init_layout)
        layout.addWidget(init_state_group)
        
        # Reference state group (compact, one line)
        ref_state_group = QGroupBox("Reference State")
        ref_layout = QHBoxLayout()
        ref_layout.setContentsMargins(10, 5, 10, 5)
        
        ref_layout.addWidget(QLabel("X (m):"))
        self.spin_x_ref = QDoubleSpinBox()
        self.spin_x_ref.setRange(-10.0, 10.0)
        self.spin_x_ref.setValue(0.0)
        self.spin_x_ref.setDecimals(1)
        self.spin_x_ref.setMaximumWidth(100)
        ref_layout.addWidget(self.spin_x_ref)
        
        ref_layout.addWidget(QLabel("Y (m):"))
        self.spin_y_ref = QDoubleSpinBox()
        self.spin_y_ref.setRange(-10.0, 10.0)
        self.spin_y_ref.setValue(0.0)
        self.spin_y_ref.setDecimals(1)
        self.spin_y_ref.setMaximumWidth(100)
        ref_layout.addWidget(self.spin_y_ref)
        
        ref_layout.addWidget(QLabel("Z (m):"))
        self.spin_z_ref = QDoubleSpinBox()
        self.spin_z_ref.setRange(0.0, 10.0)  # ENU: z >= 0 (above ground)
        self.spin_z_ref.setValue(1.0)  # 1m above ground in ENU
        self.spin_z_ref.setDecimals(1)
        self.spin_z_ref.setMaximumWidth(100)
        ref_layout.addWidget(self.spin_z_ref)
        
        ref_layout.addStretch()
        ref_state_group.setLayout(ref_layout)
        layout.addWidget(ref_state_group)
        
        # Attitude Control Debug group
        att_debug_group = QGroupBox("Attitude Control Debug")
        att_debug_layout = QVBoxLayout()
        att_debug_layout.setContentsMargins(10, 5, 10, 5)
        att_debug_layout.setSpacing(5)
        
        # Initial attitude (compact, one line)
        init_att_layout = QHBoxLayout()
        init_att_layout.addWidget(QLabel("Initial Attitude:"))
        init_att_layout.addWidget(QLabel("Roll (deg):"))
        self.spin_roll0 = QDoubleSpinBox()
        self.spin_roll0.setRange(-180.0, 180.0)
        self.spin_roll0.setValue(0.0)
        self.spin_roll0.setDecimals(1)
        self.spin_roll0.setMaximumWidth(80)
        init_att_layout.addWidget(self.spin_roll0)
        
        init_att_layout.addWidget(QLabel("Pitch (deg):"))
        self.spin_pitch0 = QDoubleSpinBox()
        self.spin_pitch0.setRange(-180.0, 180.0)
        self.spin_pitch0.setValue(0.0)
        self.spin_pitch0.setDecimals(1)
        self.spin_pitch0.setMaximumWidth(80)
        init_att_layout.addWidget(self.spin_pitch0)
        
        init_att_layout.addWidget(QLabel("Yaw (deg):"))
        self.spin_yaw0 = QDoubleSpinBox()
        self.spin_yaw0.setRange(-180.0, 180.0)
        self.spin_yaw0.setValue(0.0)
        self.spin_yaw0.setDecimals(1)
        self.spin_yaw0.setMaximumWidth(80)
        init_att_layout.addWidget(self.spin_yaw0)
        init_att_layout.addStretch()
        att_debug_layout.addLayout(init_att_layout)
        
        # Reference attitude (compact, one line)
        ref_att_layout = QHBoxLayout()
        ref_att_layout.addWidget(QLabel("Reference Attitude:"))
        ref_att_layout.addWidget(QLabel("Roll (deg):"))
        self.spin_roll_ref = QDoubleSpinBox()
        self.spin_roll_ref.setRange(-180.0, 180.0)
        self.spin_roll_ref.setValue(10.0)
        self.spin_roll_ref.setDecimals(1)
        self.spin_roll_ref.setMaximumWidth(80)
        ref_att_layout.addWidget(self.spin_roll_ref)
        
        ref_att_layout.addWidget(QLabel("Pitch (deg):"))
        self.spin_pitch_ref = QDoubleSpinBox()
        self.spin_pitch_ref.setRange(-180.0, 180.0)
        self.spin_pitch_ref.setValue(0.0)
        self.spin_pitch_ref.setDecimals(1)
        self.spin_pitch_ref.setMaximumWidth(80)
        ref_att_layout.addWidget(self.spin_pitch_ref)
        
        ref_att_layout.addWidget(QLabel("Yaw (deg):"))
        self.spin_yaw_ref = QDoubleSpinBox()
        self.spin_yaw_ref.setRange(-180.0, 180.0)
        self.spin_yaw_ref.setValue(0.0)
        self.spin_yaw_ref.setDecimals(1)
        self.spin_yaw_ref.setMaximumWidth(80)
        ref_att_layout.addWidget(self.spin_yaw_ref)
        ref_att_layout.addStretch()
        att_debug_layout.addLayout(ref_att_layout)
        
        # Run attitude debug button
        self.btn_run_att_debug = QPushButton("Run Attitude Debug")
        self.btn_run_att_debug.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_run_att_debug.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        self.btn_run_att_debug.clicked.connect(self.run_attitude_debug)
        att_debug_layout.addWidget(self.btn_run_att_debug)
        
        att_debug_group.setLayout(att_debug_layout)
        layout.addWidget(att_debug_group)
        
        # Run button
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.btn_run.clicked.connect(self.run_simulation)
        layout.addWidget(self.btn_run)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status log
        status_label = QLabel("Status Log:")
        status_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(status_label)
        
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(120)
        layout.addWidget(self.status_log)
        
        # Performance metrics table
        metrics_label = QLabel("Performance Metrics:")
        metrics_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(metrics_label)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels([
            "Controller", "Total RMSE", "Pos RMSE", "Vel RMSE", 
            "Att RMSE", "Control Effort"
        ])
        self.metrics_table.setMaximumHeight(150)
        layout.addWidget(self.metrics_table)
        
        layout.addStretch()
        
        return panel
    
    def create_result_panel(self):
        """Create the result panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        panel.setLayout(layout)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # State trajectories tab
        self.plot_states = MatplotlibWidget()
        self.tab_widget.addTab(self.plot_states, "State Trajectories")
        
        # Control inputs tab
        self.plot_controls = MatplotlibWidget()
        self.tab_widget.addTab(self.plot_controls, "Control Inputs")
        
        # Errors tab
        self.plot_errors = MatplotlibWidget()
        self.tab_widget.addTab(self.plot_errors, "Tracking Errors")
        
        # 3D trajectory tab
        self.plot_3d = MatplotlibWidget()
        self.tab_widget.addTab(self.plot_3d, "3D Trajectory")
        
        # Attitude Control Debug tab
        self.plot_att_debug = MatplotlibWidget()
        self.tab_widget.addTab(self.plot_att_debug, "Attitude Debug")
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    def init_simulator(self):
        """Initialize the simulator with default parameters"""
        phy_params = PhyParams(
            MASS=0.6570,
            G=9.81,
            I_XX=0.062796,
            I_YY=0.062976,
            I_ZZ=0.001403,
            DIST_COM_2_THRUST=0.5693,
        )
        self.simulator = ComparisonSimulator(phy_params, dt=0.01)
        self.log_status("Simulator initialized with default parameters")
    
    def update_controller_parameters(self):
        """Update controller parameters from GUI inputs"""
        # Update LQR Full-State controller
        if hasattr(self, 'lqr_full_Q'):
            Q_diag = np.array([self.lqr_full_Q[i].value() for i in range(12)])
            R_diag = np.array([self.lqr_full_R[i].value() for i in range(4)])
            Q = np.diag(Q_diag)
            R = np.diag(R_diag)
            self.simulator.lqr_full = LQRFullStateController(
                self.simulator.params, Q=Q, R=R
            )
            self.log_status("Updated Full-State LQR parameters")
        
        # Update LQR Attitude-Only controller
        if hasattr(self, 'lqr_att_Q'):
            Q_att_diag = np.array([self.lqr_att_Q[i].value() for i in range(6)])
            R_att_diag = np.array([self.lqr_att_R[i].value() for i in range(3)])
            Q_att = np.diag(Q_att_diag)
            R_att = np.diag(R_att_diag)
            self.simulator.lqr_att = LQRAttitudeOnlyController(
                self.simulator.params, Q_att=Q_att, R_att=R_att
            )
            self.log_status("Updated Attitude-Only LQR parameters")
        
        # Update PID controller
        if hasattr(self, 'pid_Kp_pos'):
            gains = {
                'Kp_pos': np.array([self.pid_Kp_pos[i].value() for i in range(3)]),
                'Ki_pos': np.array([self.pid_Ki_pos[i].value() for i in range(3)]),
                'Kd_pos': np.array([self.pid_Kd_pos[i].value() for i in range(3)]),
                'Kp_vel': np.array([self.pid_Kp_vel[i].value() for i in range(3)]),
                'Ki_vel': np.array([self.pid_Ki_vel[i].value() for i in range(3)]),
                'Kd_vel': np.array([self.pid_Kd_vel[i].value() for i in range(3)]),
                'Kp_att': np.array([self.pid_Kp_att[i].value() for i in range(3)]),
                'Ki_att': np.array([self.pid_Ki_att[i].value() for i in range(3)]),
                'Kd_att': np.array([self.pid_Kd_att[i].value() for i in range(3)]),
                'Kp_omega': np.array([self.pid_Kp_omega[i].value() for i in range(3)]),
                'Ki_omega': np.array([self.pid_Ki_omega[i].value() for i in range(3)]),
                'Kd_omega': np.array([self.pid_Kd_omega[i].value() for i in range(3)]),
            }
            self.simulator.pid.set_gains(gains)
            self.log_status("Updated PID controller parameters")
    
    def get_selected_controllers(self):
        """Get list of selected controllers"""
        controllers = []
        if self.checkbox_lqr_full.isChecked():
            controllers.append('lqr_full')
        if self.checkbox_lqr_att.isChecked():
            controllers.append('lqr_attitude')
        if self.checkbox_pid.isChecked():
            controllers.append('pid')
        return controllers
    
    def run_simulation(self):
        """Run the simulation with selected controllers"""
        # Get selected controllers
        controllers = self.get_selected_controllers()
        if not controllers:
            QMessageBox.warning(self, "No Controller Selected", 
                              "Please select at least one controller.")
            return
        
        # Get simulation parameters
        t_end = self.spin_t_end.value()
        dt = self.spin_dt.value()
        
        # Update simulator time step
        self.simulator.dt = dt
        
        # Get initial and reference states
        state0 = np.array([
            self.spin_x0.value(), self.spin_y0.value(), self.spin_z0.value(),
            0.0, 0.0, 0.0,  # velocity
            0.0, 0.0, 0.0,  # attitude
            0.0, 0.0, 0.0   # angular velocity
        ])
        
        state_ref = np.array([
            self.spin_x_ref.value(), self.spin_y_ref.value(), self.spin_z_ref.value(),
            0.0, 0.0, 0.0,  # velocity
            0.0, 0.0, 0.0,  # attitude
            0.0, 0.0, 0.0   # angular velocity
        ])
        
        # Update controller parameters from GUI before running
        self.update_controller_parameters()
        
        # Disable run button and show progress
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_status(f"Starting simulation with controllers: {', '.join(controllers)}")
        self.log_status(f"Simulation time: {t_end} s, Time step: {dt} s")
        
        # Create and start worker thread
        self.worker = SimulationWorker(self.simulator, controllers, state0, state_ref, t_end)
        self.worker.progress.connect(self.log_status)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()
    
    def on_simulation_finished(self, results):
        """Handle simulation completion"""
        self.results = results
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.log_status("Simulation completed successfully!")
        
        # Update plots
        self.update_plots()
        
        # Update metrics table
        self.update_metrics()
    
    def on_simulation_error(self, error_msg):
        """Handle simulation error"""
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.log_status(f"Error: {error_msg}")
        QMessageBox.critical(self, "Simulation Error", f"An error occurred:\n{error_msg}")
    
    def run_attitude_debug(self):
        """Run attitude control debug simulation"""
        # Get attitude parameters
        att_init_euler = np.deg2rad(np.array([
            self.spin_roll0.value(),
            self.spin_pitch0.value(),
            self.spin_yaw0.value()
        ]))
        
        att_ref_euler = np.deg2rad(np.array([
            self.spin_roll_ref.value(),
            self.spin_pitch_ref.value(),
            self.spin_yaw_ref.value()
        ]))
        
        # Get simulation parameters
        sim_time = self.spin_t_end.value()
        dt = self.spin_dt.value()
        
        # Disable button and show progress
        self.btn_run_att_debug.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_status(f"Starting attitude control debug...")
        self.log_status(f"Initial: Roll={self.spin_roll0.value():.1f}°, "
                       f"Pitch={self.spin_pitch0.value():.1f}°, Yaw={self.spin_yaw0.value():.1f}°")
        self.log_status(f"Reference: Roll={self.spin_roll_ref.value():.1f}°, "
                       f"Pitch={self.spin_pitch_ref.value():.1f}°, Yaw={self.spin_yaw_ref.value():.1f}°")
        
        # Create and start worker thread
        self.att_debug_worker = AttitudeDebugWorker(
            self.simulator.params, att_init_euler, att_ref_euler, sim_time, dt
        )
        self.att_debug_worker.progress.connect(self.log_status)
        self.att_debug_worker.finished.connect(self.on_attitude_debug_finished)
        self.att_debug_worker.error.connect(self.on_attitude_debug_error)
        self.att_debug_worker.start()
    
    def on_attitude_debug_finished(self, result):
        """Handle attitude debug simulation finished"""
        self.btn_run_att_debug.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.log_status("Attitude control debug simulation completed!")
        
        # Switch to attitude debug tab and plot results
        self.tab_widget.setCurrentIndex(4)  # Attitude Debug tab
        self.plot_att_debug.plot_attitude_debug(result)
    
    def on_attitude_debug_error(self, error_msg):
        """Handle attitude debug simulation error"""
        self.btn_run_att_debug.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.log_status(f"Error: {error_msg}")
        QMessageBox.critical(self, "Attitude Debug Error", f"An error occurred:\n{error_msg}")
    
    def update_plots(self):
        """Update all plots with simulation results"""
        if not self.results:
            return
        
        # Update state trajectories
        self.plot_states.plot_states(self.results)
        
        # Update control inputs
        self.plot_controls.plot_controls(self.results)
        
        # Update errors
        self.plot_errors.plot_errors(self.results)
        
        # Update 3D trajectory
        self.plot_3d.plot_3d_trajectory(self.results)
    
    def update_metrics(self):
        """Update metrics table"""
        if not self.results:
            return
        
        # Compute metrics
        metrics = self.simulator.compute_metrics(self.results)
        
        # Update table
        self.metrics_table.setRowCount(len(metrics))
        
        for row, (name, m) in enumerate(metrics.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{m['rmse_total']:.4f}"))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{m['rmse_pos']:.4f}"))
            self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{m['rmse_vel']:.4f}"))
            self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{m['rmse_att']:.4f}"))
            self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{m['control_effort_total']:.2f}"))
        
        self.metrics_table.resizeColumnsToContents()
    
    def log_status(self, message):
        """Add message to status log"""
        self.status_log.append(message)
        scrollbar = self.status_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main():
    """Main function to run the GUI"""
    # Enable high DPI scaling BEFORE creating QApplication
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = ControllerComparisonGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
