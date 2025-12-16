#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5 GUI for TVC LQR Controller Simulation
Provides a graphical interface for running single simulations or comparing multiple simulation modes.
"""

import sys
import os
import numpy as np

# Set environment variables for high DPI scaling before importing Qt
# This must be done before creating QApplication
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'
os.environ['QT_SCALE_FACTOR'] = '1'

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QCheckBox, QLabel, QGroupBox, QRadioButton,
                             QTextEdit, QProgressBar, QMessageBox, QTabWidget, QSpinBox,
                             QDoubleSpinBox, QFormLayout, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
# Set matplotlib DPI for high resolution displays
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 100
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from tvc_lqr_all_lei import (
    LQRController, PhyParams, SimulationMode,
    load_parameters, setup_lqr_controller,
    simulate_closed_loop, run_comparison_simulations,
    plot_simulation_results, plot_comparison_results
)


class SimulationWorker(QThread):
    """Worker thread for running simulations without blocking the GUI"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, lqr, x_ref, phy_params, constraints, modes, is_comparison=False):
        super().__init__()
        self.lqr = lqr
        self.x_ref = x_ref
        self.phy_params = phy_params
        self.constraints = constraints
        self.modes = modes
        self.is_comparison = is_comparison
        self.t_span = (0.0, 10.0)  # Will be updated from GUI
        self.n_points = 500  # Will be updated from GUI
        
    def run(self):
        try:
            if self.is_comparison:
                self.progress.emit(f"Running {len(self.modes)} modes for comparison...")
                results = run_comparison_simulations(
                    self.lqr, self.x_ref, self.phy_params, self.constraints,
                    x0=np.zeros(12),
                    t_span=self.t_span,
                    n_points=self.n_points,
                    modes=self.modes
                )
                self.finished.emit({'type': 'comparison', 'results': results})
            else:
                mode = self.modes[0]
                self.progress.emit(f"Running mode: {mode.value}")
                sol, x_traj, e_traj, u_traj, u_traj_limited, u_actuator = simulate_closed_loop(
                    self.lqr, self.x_ref, self.phy_params, self.constraints,
                    x0=np.zeros(12),
                    t_span=self.t_span,
                    n_points=self.n_points,
                    simulation_mode=mode
                )
                result = {
                    'sol': sol,
                    'x_traj': x_traj,
                    'e_traj': e_traj,
                    'u_traj': u_traj,
                    'u_traj_limited': u_traj_limited,
                    'u_actuator': u_actuator,
                    'mode': mode
                }
                self.finished.emit({'type': 'single', 'result': result})
        except Exception as e:
            self.error.emit(str(e))


class MatplotlibWidget(QWidget):
    """Widget for displaying matplotlib figures"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Optimized figure size for 1920x1080 screen
        # Right panel is ~1200px wide, so figure should be ~14-15 inches
        # DPI is set globally, so figure size in inches will scale correctly
        self.figure = Figure(figsize=(14, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_single(self, sol, x_traj, x_ref, u_traj, u_traj_limited, phy_params, constraints, u_actuator):
        """Plot single simulation results"""
        # Close any existing figures to avoid memory issues
        plt.close('all')
        
        # Call plotting function which will create a new figure
        fig = plot_simulation_results(sol, x_traj, x_ref, u_traj, u_traj_limited, 
                                     phy_params, constraints, u_actuator, show_plot=False)
        
        # Replace our figure with the new one
        if fig is not None:
            # Remove old canvas
            old_canvas = self.canvas
            self.layout().removeWidget(old_canvas)
            old_canvas.deleteLater()
            
            # Create new canvas with the figure
            self.figure = fig
            self.canvas = FigureCanvas(self.figure)
            self.layout().addWidget(self.canvas)
    
    def plot_comparison(self, results, x_ref, phy_params, constraints):
        """Plot comparison results"""
        # Close any existing figures to avoid memory issues
        plt.close('all')
        
        # Call plotting function which will create a new figure
        fig = plot_comparison_results(results, x_ref, phy_params, constraints, show_plot=False)
        
        # Replace our figure with the new one
        if fig is not None:
            # Remove old canvas
            old_canvas = self.canvas
            self.layout().removeWidget(old_canvas)
            old_canvas.deleteLater()
            
            # Create new canvas with the figure
            self.figure = fig
            self.canvas = FigureCanvas(self.figure)
            self.layout().addWidget(self.canvas)


class TVCSimulationGUI(QMainWindow):
    """Main GUI window for TVC simulation"""
    
    def __init__(self):
        super().__init__()
        self.lqr = None
        self.x_ref = None
        self.phy_params = None
        self.constraints = None
        self.worker = None
        
        self.init_ui()
        self.load_parameters()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("TVC LQR Controller Simulation")
        
        # Get screen geometry for proper scaling
        app = QApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
            screen_geometry = screen.availableGeometry()
            screen_width = screen_geometry.width()
            screen_height = screen_geometry.height()
            device_pixel_ratio = screen.devicePixelRatio()
            
            # Optimize for 1920x1080 screen (with some margin)
            # On high DPI screens, Qt will handle scaling automatically
            base_width = 1920
            base_height = 1080
            
            # Calculate window size based on base resolution
            # Use 95% of screen size to leave some margin
            if screen_width >= base_width and screen_height >= base_height:
                # Standard or larger screen
                window_width = int(screen_width * 0.95)
                window_height = int(screen_height * 0.95)
            else:
                # Smaller screen, use smaller window
                window_width = int(screen_width * 0.9)
                window_height = int(screen_height * 0.9)
            
            # Center window on screen
            x_pos = (screen_width - window_width) // 2
            y_pos = (screen_height - window_height) // 2
            
            self.setGeometry(x_pos, y_pos, window_width, window_height)
        else:
            # Fallback if QApplication is not available
            self.setGeometry(50, 50, 1900, 1000)
        
        # Set minimum size (will be scaled on high DPI screens)
        min_width = 1600
        min_height = 900
        self.setMinimumSize(min_width, min_height)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        central_widget.setLayout(main_layout)
        
        # Left panel for controls (optimized for 1920x1080)
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)  # Reduced ratio for better balance
        
        # Right panel for results
        right_panel = self.create_result_panel()
        main_layout.addWidget(right_panel, 2)  # More space for plots
        
    def create_control_panel(self):
        """Create the control panel"""
        # Create scroll area for the control panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        panel.setLayout(layout)
        # Optimized width for 1920x1080: ~600px for left panel
        panel.setMinimumWidth(550)
        panel.setMaximumWidth(650)
        
        # Title
        title = QLabel("TVC Simulation Control")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(4)
        
        # Create tab widget for single and compare modes
        self.tab_widget = QTabWidget()
        
        # Single simulation tab
        single_tab = self.create_single_simulation_tab()
        self.tab_widget.addTab(single_tab, "Single Simulation")
        
        # Compare modes tab
        compare_tab = self.create_compare_modes_tab()
        self.tab_widget.addTab(compare_tab, "Compare Modes")
        
        layout.addWidget(self.tab_widget)
        layout.addSpacing(4)
        
        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        self.run_button = QPushButton("Run Simulation")
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; font-size: 11px;")
        self.run_button.setMinimumHeight(35)
        self.run_button.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_button)
        
        self.clear_button = QPushButton("Clear Results")
        self.clear_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px; font-size: 11px;")
        self.clear_button.setMinimumHeight(35)
        self.clear_button.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        layout.addSpacing(4)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(22)
        layout.addWidget(self.progress_bar)
        
        # Status text
        status_label = QLabel("Status:")
        status_label.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(status_label)
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        self.status_text.setFont(QFont("Courier", 8))
        layout.addWidget(self.status_text)
        layout.addSpacing(4)
        
        # Parameters group (shared by both tabs)
        params_group = QGroupBox("Simulation Parameters")
        params_group.setFont(QFont("Arial", 9, QFont.Bold))
        params_layout = QFormLayout()
        params_layout.setSpacing(6)
        params_layout.setVerticalSpacing(4)
        
        self.time_span_start = QDoubleSpinBox()
        self.time_span_start.setRange(0.0, 100.0)
        self.time_span_start.setValue(0.0)
        self.time_span_start.setDecimals(1)
        
        self.time_span_end = QDoubleSpinBox()
        self.time_span_end.setRange(0.1, 100.0)
        self.time_span_end.setValue(10.0)
        self.time_span_end.setDecimals(1)
        
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(50, 5000)
        self.n_points_spin.setValue(500)
        
        # Target position inputs
        self.target_x = QDoubleSpinBox()
        self.target_x.setRange(-100.0, 100.0)
        self.target_x.setValue(3.0)
        self.target_x.setDecimals(2)
        
        self.target_y = QDoubleSpinBox()
        self.target_y.setRange(-100.0, 100.0)
        self.target_y.setValue(3.0)
        self.target_y.setDecimals(2)
        
        self.target_z = QDoubleSpinBox()
        self.target_z.setRange(-100.0, 100.0)
        self.target_z.setValue(3.0)
        self.target_z.setDecimals(2)
        
        params_layout.addRow("Time Start (s):", self.time_span_start)
        params_layout.addRow("Time End (s):", self.time_span_end)
        params_layout.addRow("Number of Points:", self.n_points_spin)
        params_layout.addRow("Target X (m):", self.target_x)
        params_layout.addRow("Target Y (m):", self.target_y)
        params_layout.addRow("Target Z (m):", self.target_z)
        
        # Connect target position changes to update x_ref
        self.target_x.valueChanged.connect(self.update_target_position)
        self.target_y.valueChanged.connect(self.update_target_position)
        self.target_z.valueChanged.connect(self.update_target_position)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Error Limitations group
        error_limits_group = QGroupBox("Error Limitations")
        error_limits_group.setFont(QFont("Arial", 9, QFont.Bold))
        error_limits_layout = QFormLayout()
        error_limits_layout.setSpacing(6)
        error_limits_layout.setVerticalSpacing(4)
        
        self.error_pos_max = QDoubleSpinBox()
        self.error_pos_max.setRange(0.01, 10.0)
        self.error_pos_max.setValue(1.0)
        self.error_pos_max.setDecimals(2)
        
        self.error_vel_max = QDoubleSpinBox()
        self.error_vel_max.setRange(0.01, 50.0)
        self.error_vel_max.setValue(5.0)
        self.error_vel_max.setDecimals(2)
        
        self.error_att_max = QDoubleSpinBox()
        self.error_att_max.setRange(0.01, 5.0)
        self.error_att_max.setValue(0.5)
        self.error_att_max.setDecimals(2)
        
        self.error_angvel_max = QDoubleSpinBox()
        self.error_angvel_max.setRange(0.01, 20.0)
        self.error_angvel_max.setValue(2.0)
        self.error_angvel_max.setDecimals(2)
        
        error_limits_layout.addRow("Position Error Max (m):", self.error_pos_max)
        error_limits_layout.addRow("Velocity Error Max (m/s):", self.error_vel_max)
        error_limits_layout.addRow("Attitude Error Max:", self.error_att_max)
        error_limits_layout.addRow("Angular Vel Error Max (rad/s):", self.error_angvel_max)
        
        error_limits_group.setLayout(error_limits_layout)
        layout.addWidget(error_limits_group)
        
        # Actuator Limitations group
        actuator_limits_group = QGroupBox("Actuator Limitations")
        actuator_limits_group.setFont(QFont("Arial", 9, QFont.Bold))
        actuator_limits_layout = QFormLayout()
        actuator_limits_layout.setSpacing(6)
        actuator_limits_layout.setVerticalSpacing(4)
        
        # Servo 0 (symmetric, only max needed)
        self.servo_0_max = QDoubleSpinBox()
        self.servo_0_max.setRange(0.0, 1.0)
        self.servo_0_max.setValue(0.15)
        self.servo_0_max.setDecimals(3)
        
        # Servo 1 (symmetric, only max needed)
        self.servo_1_max = QDoubleSpinBox()
        self.servo_1_max.setRange(0.0, 1.0)
        self.servo_1_max.setValue(0.15)
        self.servo_1_max.setDecimals(3)
        
        # Thrust Total (asymmetric, both min and max needed)
        self.thrust_total_min = QDoubleSpinBox()
        self.thrust_total_min.setRange(0.0, 100.0)
        self.thrust_total_min.setValue(0.0)
        self.thrust_total_min.setDecimals(2)
        
        self.thrust_total_max = QDoubleSpinBox()
        self.thrust_total_max.setRange(1.0, 200.0)
        # Default value will be set after phy_params is loaded
        self.thrust_total_max.setValue(9.67)  # Temporary default: 1.5 * 0.657 * 9.81
        self.thrust_total_max.setDecimals(2)
        
        # Yaw Torque (symmetric, only max needed)
        self.r_cmd_max = QDoubleSpinBox()
        self.r_cmd_max.setRange(0.0, 5.0)
        self.r_cmd_max.setValue(0.5)
        self.r_cmd_max.setDecimals(2)
        
        actuator_limits_layout.addRow("Servo 0 Max (rad):", self.servo_0_max)
        actuator_limits_layout.addRow("Servo 1 Max (rad):", self.servo_1_max)
        actuator_limits_layout.addRow("Thrust Total Min (N):", self.thrust_total_min)
        actuator_limits_layout.addRow("Thrust Total Max (N):", self.thrust_total_max)
        actuator_limits_layout.addRow("Yaw Torque Max (Nm):", self.r_cmd_max)
        
        actuator_limits_group.setLayout(actuator_limits_layout)
        layout.addWidget(actuator_limits_group)
        
        # Actuator Dynamics group
        actuator_dynamics_group = QGroupBox("Actuator Dynamics")
        actuator_dynamics_group.setFont(QFont("Arial", 9, QFont.Bold))
        actuator_dynamics_layout = QFormLayout()
        actuator_dynamics_layout.setSpacing(6)
        actuator_dynamics_layout.setVerticalSpacing(4)
        
        self.tau_thrust_angle = QDoubleSpinBox()
        self.tau_thrust_angle.setRange(0.001, 10.0)
        self.tau_thrust_angle.setValue(0.1)
        self.tau_thrust_angle.setDecimals(3)
        
        self.tau_thrust = QDoubleSpinBox()
        self.tau_thrust.setRange(0.001, 10.0)
        self.tau_thrust.setValue(0.5)
        self.tau_thrust.setDecimals(3)
        
        self.tau_yaw_torque = QDoubleSpinBox()
        self.tau_yaw_torque.setRange(0.001, 10.0)
        self.tau_yaw_torque.setValue(0.05)
        self.tau_yaw_torque.setDecimals(3)
        
        actuator_dynamics_layout.addRow("Thrust Angle τ (s):", self.tau_thrust_angle)
        actuator_dynamics_layout.addRow("Thrust τ (s):", self.tau_thrust)
        actuator_dynamics_layout.addRow("Yaw Torque τ (s):", self.tau_yaw_torque)
        
        actuator_dynamics_group.setLayout(actuator_dynamics_layout)
        layout.addWidget(actuator_dynamics_group)
        
        layout.addStretch()
        
        # Set the panel as scroll area widget
        scroll.setWidget(panel)
        return scroll
    
    def create_single_simulation_tab(self):
        """Create the single simulation tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)
        tab.setLayout(layout)
        
        # Mode selection group
        mode_group = QGroupBox("Select Simulation Mode")
        mode_group.setFont(QFont("Arial", 9, QFont.Bold))
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(4)
        
        # Mode checkboxes (single selection)
        self.single_mode_checkboxes = {}
        mode_list = list(SimulationMode)
        for mode in mode_list:
            checkbox = QCheckBox(mode.value)
            checkbox.setChecked(False)
            self.single_mode_checkboxes[mode] = checkbox
            checkbox.stateChanged.connect(lambda state, m=mode: self.on_single_checkbox_changed(state, m))
            mode_layout.addWidget(checkbox)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        layout.addStretch()
        return tab
    
    def create_compare_modes_tab(self):
        """Create the compare modes tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)
        tab.setLayout(layout)
        
        # Mode selection group
        mode_group = QGroupBox("Select Modes to Compare (at least 2)")
        mode_group.setFont(QFont("Arial", 9, QFont.Bold))
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(4)
        
        # Mode checkboxes (multiple selection)
        self.compare_mode_checkboxes = {}
        mode_list = list(SimulationMode)
        for mode in mode_list:
            checkbox = QCheckBox(mode.value)
            checkbox.setChecked(False)
            self.compare_mode_checkboxes[mode] = checkbox
            mode_layout.addWidget(checkbox)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        layout.addStretch()
        return tab
    
    def on_single_checkbox_changed(self, state, mode):
        """Handle checkbox change in single mode"""
        if state == Qt.Checked:
            # Uncheck all other checkboxes
            for m, checkbox in self.single_mode_checkboxes.items():
                if m != mode:
                    checkbox.setChecked(False)
    
    def create_result_panel(self):
        """Create the result display panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)
        panel.setLayout(layout)
        
        # Title
        title = QLabel("Simulation Results")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Matplotlib widget
        self.plot_widget = MatplotlibWidget()
        layout.addWidget(self.plot_widget)
        
        return panel
    
    def update_target_position(self):
        """Update target position in x_ref"""
        if self.x_ref is not None:
            target_position = np.array([
                self.target_x.value(),
                self.target_y.value(),
                self.target_z.value()
            ])
            target_velocity = np.array([0.0, 0.0, 0.0])
            target_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            target_angular_velocity = np.array([0.0, 0.0, 0.0])
            
            self.x_ref = np.concatenate([
                target_position,
                target_velocity,
                target_quaternion[:3],
                target_angular_velocity
            ])
            self.log_status(f"Target position updated: ({target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f})")
    
    def update_constraints(self):
        """Update constraints from GUI inputs"""
        if self.constraints is None:
            return
        
        # Update error limits
        error_pos_max_val = self.error_pos_max.value()
        error_vel_max_val = self.error_vel_max.value()
        error_att_max_val = self.error_att_max.value()
        error_angvel_max_val = self.error_angvel_max.value()
        
        self.constraints['error_pos_max'] = np.array([error_pos_max_val, error_pos_max_val, error_pos_max_val])
        self.constraints['error_vel_max'] = np.array([error_vel_max_val, error_vel_max_val, error_vel_max_val])
        self.constraints['error_att_max'] = np.array([error_att_max_val, error_att_max_val, error_att_max_val])
        self.constraints['error_angvel_max'] = np.array([error_angvel_max_val, error_angvel_max_val, error_angvel_max_val])
        
        # Update actuator limitations
        # Servo 0 and Servo 1 are symmetric (min = -max)
        servo_0_max = self.servo_0_max.value()
        self.constraints['servo_0_min'] = -servo_0_max
        self.constraints['servo_0_max'] = servo_0_max
        
        servo_1_max = self.servo_1_max.value()
        self.constraints['servo_1_min'] = -servo_1_max
        self.constraints['servo_1_max'] = servo_1_max
        
        # Thrust Total is asymmetric (both min and max needed)
        self.constraints['thrust_total_min'] = self.thrust_total_min.value()
        self.constraints['thrust_total_max'] = self.thrust_total_max.value()
        
        # Yaw Torque is symmetric (min = -max)
        r_cmd_max = self.r_cmd_max.value()
        self.constraints['r_cmd_min'] = -r_cmd_max
        self.constraints['r_cmd_max'] = r_cmd_max
        
        # Update actuator dynamics
        self.constraints['tau_thrust_angle'] = self.tau_thrust_angle.value()
        self.constraints['tau_thrust'] = self.tau_thrust.value()
        self.constraints['tau_yaw_torque'] = self.tau_yaw_torque.value()
        
        # Calculate thrust_cmd limits from thrust_total limits
        if self.phy_params is not None:
            self.constraints['thrust_cmd_min'] = self.constraints['thrust_total_min'] - self.phy_params.MASS * self.phy_params.G
            self.constraints['thrust_cmd_max'] = self.constraints['thrust_total_max'] - self.phy_params.MASS * self.phy_params.G
    
    def load_parameters(self):
        """Load simulation parameters"""
        try:
            self.phy_params, Q_matrix, R_matrix, self.constraints = load_parameters()
            
            # Set default constraints if not present
            if 'servo_0_min' not in self.constraints:
                self.constraints['servo_0_min'] = -0.15
                self.constraints['servo_0_max'] = 0.15
                self.constraints['servo_1_min'] = -0.15
                self.constraints['servo_1_max'] = 0.15
                self.constraints['thrust_total_min'] = 0.0
                self.constraints['thrust_total_max'] = 1.5 * self.phy_params.MASS * self.phy_params.G
                self.constraints['r_cmd_min'] = -0.5
                self.constraints['r_cmd_max'] = 0.5
            
            # Set error limits and actuator parameters (defaults, will be updated from GUI)
            self.constraints['error_pos_max'] = np.array([1.0, 1.0, 1.0])
            self.constraints['error_vel_max'] = np.array([5.0, 5.0, 5.0])
            self.constraints['error_att_max'] = np.array([0.5, 0.5, 0.5])
            self.constraints['error_angvel_max'] = np.array([2.0, 2.0, 2.0])
            self.constraints['tau_thrust_angle'] = 0.1
            self.constraints['tau_thrust'] = 0.5
            self.constraints['tau_yaw_torque'] = 0.05
            
            # Update GUI default values from loaded constraints (if widgets exist)
            if hasattr(self, 'error_pos_max'):
                # Update GUI values to match loaded constraints
                self.error_pos_max.setValue(self.constraints['error_pos_max'][0])
                self.error_vel_max.setValue(self.constraints['error_vel_max'][0])
                self.error_att_max.setValue(self.constraints['error_att_max'][0])
                self.error_angvel_max.setValue(self.constraints['error_angvel_max'][0])
                
                # Servo 0 and Servo 1: only set max (min is automatically -max)
                self.servo_0_max.setValue(self.constraints['servo_0_max'])
                self.servo_1_max.setValue(self.constraints['servo_1_max'])
                
                # Thrust Total: set both min and max
                self.thrust_total_min.setValue(self.constraints['thrust_total_min'])
                self.thrust_total_max.setValue(self.constraints['thrust_total_max'])
                
                # Yaw Torque: only set max (min is automatically -max)
                self.r_cmd_max.setValue(self.constraints['r_cmd_max'])
                
                self.tau_thrust_angle.setValue(self.constraints['tau_thrust_angle'])
                self.tau_thrust.setValue(self.constraints['tau_thrust'])
                self.tau_yaw_torque.setValue(self.constraints['tau_yaw_torque'])
                
                # Now update constraints from GUI to ensure consistency
                self.update_constraints()
            
            # Set reference state (will be updated by update_target_position)
            target_position = np.array([3.0, 3.0, 3.0])  # Default values
            target_velocity = np.array([0.0, 0.0, 0.0])
            target_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            target_angular_velocity = np.array([0.0, 0.0, 0.0])
            
            self.x_ref = np.concatenate([
                target_position,
                target_velocity,
                target_quaternion[:3],
                target_angular_velocity
            ])
            
            # Update target position from GUI (if widgets exist)
            if hasattr(self, 'target_x'):
                self.update_target_position()
            
            # Setup LQR controller
            self.lqr, success = setup_lqr_controller(self.phy_params, Q_matrix, R_matrix)
            
            if not success:
                QMessageBox.critical(self, "Error", "Failed to solve LQR. Please check parameters.")
                return
            
            self.log_status("Parameters loaded successfully.")
            self.log_status(f"LQR controller initialized.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")
    
    def get_selected_modes(self):
        """Get selected simulation modes based on current tab"""
        selected_modes = []
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # Single simulation tab
            for mode, checkbox in self.single_mode_checkboxes.items():
                if checkbox.isChecked():
                    selected_modes.append(mode)
        else:  # Compare modes tab
            for mode, checkbox in self.compare_mode_checkboxes.items():
                if checkbox.isChecked():
                    selected_modes.append(mode)
        
        return selected_modes
    
    def run_simulation(self):
        """Run the simulation"""
        if self.lqr is None:
            QMessageBox.warning(self, "Warning", "Parameters not loaded. Please check initialization.")
            return
        
        # Update target position and constraints before running
        self.update_target_position()
        self.update_constraints()
        
        # Get selected modes
        selected_modes = self.get_selected_modes()
        current_tab = self.tab_widget.currentIndex()
        is_comparison = (current_tab == 1)  # Compare modes tab
        
        if len(selected_modes) == 0:
            QMessageBox.warning(self, "Warning", "Please select at least one simulation mode.")
            return
        
        if is_comparison and len(selected_modes) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 modes for comparison.")
            return
        
        if not is_comparison and len(selected_modes) > 1:
            QMessageBox.warning(self, "Warning", "Please select only one mode for single simulation.")
            return
        
        # Disable run button
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Clear previous results
        self.plot_widget.figure.clear()
        self.plot_widget.canvas.draw()
        
        # Update worker parameters from GUI
        t_span = (self.time_span_start.value(), self.time_span_end.value())
        n_points = self.n_points_spin.value()
        
        # Create and start worker thread
        self.worker = SimulationWorker(
            self.lqr, self.x_ref, self.phy_params, self.constraints,
            selected_modes, is_comparison
        )
        self.worker.t_span = t_span
        self.worker.n_points = n_points
        self.worker.progress.connect(self.log_status)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()
    
    def on_simulation_finished(self, result_dict):
        """Handle simulation completion"""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        
        try:
            if result_dict['type'] == 'comparison':
                results = result_dict['results']
                self.plot_widget.plot_comparison(results, self.x_ref, self.phy_params, self.constraints)
                self.log_status(f"Comparison completed: {len(results)} modes")
            else:
                result = result_dict['result']
                sol = result['sol']
                x_traj = result['x_traj']
                u_traj = result['u_traj']
                u_traj_limited = result['u_traj_limited']
                u_actuator = result['u_actuator']
                mode = result['mode']
                
                self.plot_widget.plot_single(sol, x_traj, self.x_ref, u_traj, 
                                            u_traj_limited, self.phy_params, 
                                            self.constraints, u_actuator)
                self.log_status(f"Simulation completed: {mode.value}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display results: {str(e)}")
            self.log_status(f"Error: {str(e)}")
    
    def on_simulation_error(self, error_msg):
        """Handle simulation error"""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "Simulation Error", error_msg)
        self.log_status(f"Error: {error_msg}")
    
    def clear_results(self):
        """Clear the result display"""
        try:
            self.plot_widget.figure.clear()
            self.plot_widget.canvas.draw()
            self.log_status("Results cleared.")
        except:
            # If figure was replaced, create a new empty one
            self.plot_widget.figure = Figure(figsize=(12, 8))
            old_canvas = self.plot_widget.canvas
            self.plot_widget.layout().removeWidget(old_canvas)
            old_canvas.deleteLater()
            self.plot_widget.canvas = FigureCanvas(self.plot_widget.figure)
            self.plot_widget.layout().addWidget(self.plot_widget.canvas)
            self.log_status("Results cleared.")
    
    def log_status(self, message):
        """Log status message"""
        self.status_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main():
    """Main function to run the GUI"""
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Enable high DPI scaling for high resolution displays (4K, Retina, etc.)
    # This ensures the GUI looks good on both 1920x1080 and high DPI screens
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Set DPI awareness (for Windows, if applicable)
    try:
        # Try to set high DPI scale factor rounding policy (Qt 5.14+)
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            # Additional scaling settings
            app.setAttribute(Qt.AA_Use96Dpi, False)  # Use system DPI instead of 96 DPI
    except:
        pass
    
    # Set application style
    app.setStyle('Fusion')
    
    # Get screen DPI for logging
    screen = app.primaryScreen()
    dpi = screen.physicalDotsPerInch()
    scale_factor = screen.devicePixelRatio()
    print(f"Screen DPI: {dpi:.1f}, Scale Factor: {scale_factor:.2f}")
    
    window = TVCSimulationGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
