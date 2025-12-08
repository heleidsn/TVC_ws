#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleStatus, VehicleOdometry, VehicleAttitude
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint, ActuatorMotors, ActuatorServos
import numpy as np
from tvc_controller.lqr import LQRController 
from typing import List, Dict, Any
import math
from nav_msgs.msg import Odometry
# import tvc_controller_msgs

"""
Coordinate Convention:
- NED (North, East, Down) frame is used for position and velocity.
- FRD (Forward, Right, Down) frame is used for angular velocity.
"""

class TVC_CONTROLLER(Node):   
    def __init__(self):
        """
        Initialize the Thrust Vector Control (TVC) controller node.
        
        This constructor sets up:
        - ROS2 node with proper QoS profiles for PX4 compatibility
        - Parameter loading from YAML configuration
        - LQR controller initialization with loaded parameters
        - Publishers for offboard control, vehicle commands, thrust and torque setpoints
        - Subscribers for vehicle odometry, status, and attitude ground truth
        - Internal state variables and timers for control loops
        
        The controller operates at 10Hz for general operations and 100Hz for control loops.
        """
        super().__init__('tvc_controller')

        # Configure QoS profile for compatibility with PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Initialize parameters
        self.load_parameters()

        # Print loaded parameters
        self.log_parameters()

        # Initialize LQR controller
        lqr_controller = LQRController(phy_params=self.get_physical_properties())
        success = lqr_controller.set_Q(self.Q_matrix)

        if not success:
            self.get_logger().error("Failed to set Q matrix for LQR controller.")
            return

        success = lqr_controller.set_R(self.R_matrix)

        if not success:
            self.get_logger().error("Failed to set R matrix for LQR controller.")
            return
        
        # Solve the LQR problem
        success = lqr_controller.solve_lqr()
        if not success:
            self.get_logger().error("Failed to solve LQR problem.")
            return

        # Get the gain matrix K
        self.K = lqr_controller.get_K()
        if self.K is None:
            self.get_logger().error("LQR gain matrix K is not defined. Please check your LQR setup.")
            return
        
        self.get_logger().info(f"LQR gain matrix K:\n{self.K}")
        self.get_logger().info(f"Size of K: {self.K.shape}")

        # Publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        if self.use_onboard_control_allocation:
            self.actuator_motor_setpoint_publisher = self.create_publisher(
                ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)
            self.actuator_servo_setpoint_publisher = self.create_publisher(
                ActuatorServos, '/fmu/in/actuator_servos', qos_profile)
        else:
            self.thrust_setpoint_publisher = self.create_publisher(
                VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
            self.torque_setpoint_publisher = self.create_publisher(
                VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        
        # Subscribers
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback, qos_profile)
        
        self.vehicle_odom_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback, qos_profile)
        
        self.vehicle_attitude_gt_subscriber = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude_groundtruth',
            self.vehicle_attitude_gt_callback, qos_profile)
        
        # self.groundtruth_odom_subscriber = self.create_subscription(
        #     Odometry, '/model/tvc_0/odometry_ned', 
        #     self.groundtruth_odometry_callback, 10
        # )

        # Variables
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_DISARMED
        
        self.is_odom_state_initialized = False
        self.is_attitude_gt_initialized = False

        self.is_final_position_state_initialized = False
        self.is_final_quaternion_state_initialized = False

        self.start_controller = False
        self.offboard_setpoint_counter = 0

        self.odom_state_callback_counter = 0
        self.quaternion_gt_callback_counter = 0
        
        # Initialize as empty lists to collect measurements
        self._list_of_positions = []
        self._list_of_quaternions = []

        self._last_update_position = None
        self._last_update_velocity = None
        self._last_update_quaternion = None
        self._last_update_angular_velocity = None
        
        # NEW: Variables for orientation normalization
        self.initial_rotation_matrix = None  # Fixed rotation matrix calculated at startup
        self.is_orientation_normalized = False

        # Timer
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.control_time = self.create_timer(0.01, self.control_timer)  # 100Hz
        self.get_logger().info('PX4 Controller initialized')

    def load_parameters(self) -> None:
        """
        Load all YAML parameters into class attributes.
        
        This method declares and loads parameters for:
        - Physical properties (mass, inertia, dimensions, gravity)
        - LQR controller weights (Q and R matrices)
        - Target position, velocity, orientation, and angular velocity
        - Actuator constraints (servo limits, thrust limits, torque limits)
        - Control allocation parameters (thrust/moment coefficients, PWM conversion)
        
        All parameters are loaded from the ROS2 parameter server and stored
        as class attributes for use throughout the controller.
        
        Raises:
            Exception: If required parameters are not found or have invalid values
        """
        
        # =====================================
        # TVC PLATFORM PHYSICAL PROPERTIES
        # =====================================
        self.declare_parameter('physical.G', 9.81)
        self.declare_parameter('physical.DIST_COM_2_THRUST', 0.5693)
        self.declare_parameter('physical.I_XX', 0.062796)
        self.declare_parameter('physical.I_YY', 0.062976)
        self.declare_parameter('physical.I_ZZ', 0.001403)
        self.declare_parameter('physical.MASS', 0.6570)
        
        # Load physical properties
        self.G = self.get_parameter('physical.G').get_parameter_value().double_value
        self.DIST_COM_2_THRUST = self.get_parameter('physical.DIST_COM_2_THRUST').get_parameter_value().double_value
        self.I_XX = self.get_parameter('physical.I_XX').get_parameter_value().double_value
        self.I_YY = self.get_parameter('physical.I_YY').get_parameter_value().double_value
        self.I_ZZ = self.get_parameter('physical.I_ZZ').get_parameter_value().double_value
        self.MASS = self.get_parameter('physical.MASS').get_parameter_value().double_value
        
        # Create inertia matrix
        self.inertia_matrix = np.diag([self.I_XX, self.I_YY, self.I_ZZ])
        
        # =====================================
        # LQR CONTROLLER WEIGHTS
        # =====================================
        
        # Declare Q diagonal parameters
        default_Q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.01]
        self.declare_parameter('lqr.Q_diagonal', default_Q)
        
        # Declare R diagonal parameters
        default_R = [10.0, 10.0, 1.0, 10.0]
        self.declare_parameter('lqr.R_diagonal', default_R)
        
        # Load LQR weights
        self.Q_diagonal = self.get_parameter('lqr.Q_diagonal').get_parameter_value().double_array_value
        self.R_diagonal = self.get_parameter('lqr.R_diagonal').get_parameter_value().double_array_value
        
        # Create Q and R matrices
        self.Q_matrix = np.diag(self.Q_diagonal)
        self.R_matrix = np.diag(self.R_diagonal)
        
        # State and control dimensions
        self.state_dim = len(self.Q_diagonal)  # 12 states
        self.control_dim = len(self.R_diagonal)  # 4 control inputs
        
        # =====================================
        # TARGET POSITION AND ORIENTATION
        # =====================================
        
        # Target position
        self.declare_parameter('target.position.x', 0.0)
        self.declare_parameter('target.position.y', 0.0)
        self.declare_parameter('target.position.z', -2.0)
        
        # Target velocity
        self.declare_parameter('target.velocity.x_dot', 0.0)
        self.declare_parameter('target.velocity.y_dot', 0.0)
        self.declare_parameter('target.velocity.z_dot', 0.0)
        
        # Target orientation (quaternion)
        # self.declare_parameter('target.orientation.q_x', 0.0)
        # self.declare_parameter('target.orientation.q_y', 0.0)
        # self.declare_parameter('target.orientation.q_z', 0.707) 
        # self.declare_parameter('target.orientation.q_w', 0.707)  
        
        self.declare_parameter('target.orientation.q_x', 0.0)
        self.declare_parameter('target.orientation.q_y', 0.0)
        self.declare_parameter('target.orientation.q_z', 0.0) 
        self.declare_parameter('target.orientation.q_w', 1.0)
        
        # Target angular velocity
        self.declare_parameter('target.angular_velocity.p', 0.0)
        self.declare_parameter('target.angular_velocity.q', 0.0)
        self.declare_parameter('target.angular_velocity.r', 0.0)
        
        # Load target values
        self.target_position = np.array([
            self.get_parameter('target.position.x').get_parameter_value().double_value,
            self.get_parameter('target.position.y').get_parameter_value().double_value,
            self.get_parameter('target.position.z').get_parameter_value().double_value
        ])
        
        self.target_velocity = np.array([
            self.get_parameter('target.velocity.x_dot').get_parameter_value().double_value,
            self.get_parameter('target.velocity.y_dot').get_parameter_value().double_value,
            self.get_parameter('target.velocity.z_dot').get_parameter_value().double_value
        ])
        
        self.target_quaternion = np.array([
            self.get_parameter('target.orientation.q_x').get_parameter_value().double_value,
            self.get_parameter('target.orientation.q_y').get_parameter_value().double_value,
            self.get_parameter('target.orientation.q_z').get_parameter_value().double_value,
            self.get_parameter('target.orientation.q_w').get_parameter_value().double_value
        ])
        
        self.target_angular_velocity = np.array([
            self.get_parameter('target.angular_velocity.p').get_parameter_value().double_value,
            self.get_parameter('target.angular_velocity.q').get_parameter_value().double_value,
            self.get_parameter('target.angular_velocity.r').get_parameter_value().double_value
        ])
        
        # Create target state vector for LQR (excluding q_w as mentioned in comments)
        # Order: [x, y, z, x_dot, y_dot, z_dot, q_x, q_y, q_z, q_w, p, q, r]
        self.target_state = np.concatenate([
            self.target_position,
            self.target_velocity,
            self.target_quaternion,  
            self.target_angular_velocity
        ])

        # =====================================
        # ACTUATOR CONSTRAINTS  
        # =====================================
        # Servo angle limits
        self.declare_parameter('actuator_constraints.servo_0.min', -0.25)
        self.declare_parameter('actuator_constraints.servo_0.max', 0.25)
        self.declare_parameter('actuator_constraints.servo_1.min', -0.25)
        self.declare_parameter('actuator_constraints.servo_1.max', 0.25)

        # Thrust limits
        self.declare_parameter('actuator_constraints.thrust.min_g_multiplier', -1.5)
        self.declare_parameter('actuator_constraints.thrust.max_weight_fraction', 0.1)

        # Torque limits
        self.declare_parameter('actuator_constraints.torque_z.min', -0.5)
        self.declare_parameter('actuator_constraints.torque_z.max', 0.5)

        # Load actuator constraints
        self.servo_0_min = self.get_parameter('actuator_constraints.servo_0.min').get_parameter_value().double_value
        self.servo_0_max = self.get_parameter('actuator_constraints.servo_0.max').get_parameter_value().double_value
        self.servo_1_min = self.get_parameter('actuator_constraints.servo_1.min').get_parameter_value().double_value
        self.servo_1_max = self.get_parameter('actuator_constraints.servo_1.max').get_parameter_value().double_value

        self.thrust_min_g_multiplier = self.get_parameter('actuator_constraints.thrust.min_g_multiplier').get_parameter_value().double_value
        self.thrust_max_weight_fraction = self.get_parameter('actuator_constraints.thrust.max_weight_fraction').get_parameter_value().double_value

        self.torque_z_min = self.get_parameter('actuator_constraints.torque_z.min').get_parameter_value().double_value
        self.torque_z_max = self.get_parameter('actuator_constraints.torque_z.max').get_parameter_value().double_value

        # =====================================
        # CONTROL ALLOCATION PARAMETERS
        # =====================================
        
        # Control allocation mode
        self.declare_parameter('control_allocation.use_in_onboard_computer', True)

        # Load control allocation parameter
        self.use_onboard_control_allocation = self.get_parameter('control_allocation.use_in_onboard_computer').get_parameter_value().bool_value

        if self.use_onboard_control_allocation:
            # Thrust coefficients
            self.declare_parameter('control_allocation.thrust_coefficients.ct_0', 0.0000048449)
            self.declare_parameter('control_allocation.thrust_coefficients.ct_1', 0.0000048449)
            
            # Moment coefficients
            self.declare_parameter('control_allocation.moment_coefficients.cm_0', 0.1)
            self.declare_parameter('control_allocation.moment_coefficients.cm_1', 0.1)
            
            # PWM conversion polynomial coefficients
            self.declare_parameter('control_allocation.pwm_conversion.p1', 3.1352e-4)
            self.declare_parameter('control_allocation.pwm_conversion.p2', 0.1352)
            self.declare_parameter('control_allocation.pwm_conversion.p3', 996.9672)
                        
            # Load thrust coefficients
            self.ct_0 = self.get_parameter('control_allocation.thrust_coefficients.ct_0').get_parameter_value().double_value
            self.ct_1 = self.get_parameter('control_allocation.thrust_coefficients.ct_1').get_parameter_value().double_value
            
            # Load moment coefficients
            self.cm_0 = self.get_parameter('control_allocation.moment_coefficients.cm_0').get_parameter_value().double_value
            self.cm_1 = self.get_parameter('control_allocation.moment_coefficients.cm_1').get_parameter_value().double_value
            
            # Load PWM conversion coefficients
            self.pwm_p1 = self.get_parameter('control_allocation.pwm_conversion.p1').get_parameter_value().double_value
            self.pwm_p2 = self.get_parameter('control_allocation.pwm_conversion.p2').get_parameter_value().double_value
            self.pwm_p3 = self.get_parameter('control_allocation.pwm_conversion.p3').get_parameter_value().double_value
        
            # Create control allocation matrices using loaded parameters
            self.create_control_allocation_matrices()      

    def log_parameters(self) -> None:
        """
        Log all loaded parameters for verification and debugging.
        
        This method outputs all configuration parameters to the ROS2 logger,
        including physical properties, LQR weights, target states, actuator
        constraints, and control allocation parameters. This is useful for 
        verifying correct parameter loading and debugging configuration issues.
        """
        self.get_logger().info("=== PX4Controller Parameters Loaded ===")
        
        # Physical properties
        self.get_logger().info(f"Physical Properties:")
        self.get_logger().info(f"  G: {self.G} m/s²")
        self.get_logger().info(f"  DIST_COM_2_THRUST: {self.DIST_COM_2_THRUST} m")
        self.get_logger().info(f"  Inertia - I_XX: {self.I_XX}, I_YY: {self.I_YY}, I_ZZ: {self.I_ZZ} kg·m²")
        self.get_logger().info(f"  MASS: {self.MASS} kg")
        
        # LQR weights
        self.get_logger().info(f"LQR Controller:")
        self.get_logger().info(f"  Q diagonal: {self.Q_diagonal}")
        self.get_logger().info(f"  R diagonal: {self.R_diagonal}")
        self.get_logger().info(f"  State dimension: {self.state_dim}")
        self.get_logger().info(f"  Control dimension: {self.control_dim}")
        
        # Target values
        self.get_logger().info(f"Target State:")
        self.get_logger().info(f"  Position: {self.target_position} m")
        self.get_logger().info(f"  Velocity: {self.target_velocity} m/s")
        self.get_logger().info(f"  Quaternion: {self.target_quaternion}")
        self.get_logger().info(f"  Angular velocity: {self.target_angular_velocity} rad/s")
        self.get_logger().info(f"  Target state vector: {self.target_state}")

        # Actuator constraints
        self.get_logger().info(f"Actuator Constraints:")
        self.get_logger().info(f"  Servo 0 limits: [{self.servo_0_min}, {self.servo_0_max}] rad")
        self.get_logger().info(f"  Servo 1 limits: [{self.servo_1_min}, {self.servo_1_max}] rad")
        self.get_logger().info(f"  Thrust min G multiplier: {self.thrust_min_g_multiplier}")
        self.get_logger().info(f"  Thrust max weight fraction: {self.thrust_max_weight_fraction}")
        self.get_logger().info(f"  Torque Z limits: [{self.torque_z_min}, {self.torque_z_max}] Nm")

        # Control allocation parameters
        self.get_logger().info(f"Run control allocation on onboard computer: {self.use_onboard_control_allocation}")
        if self.use_onboard_control_allocation:
            self.get_logger().info(f"  Use onboard computer: {self.use_onboard_control_allocation}")
            self.get_logger().info(f"  Thrust coefficients - ct_0: {self.ct_0}, ct_1: {self.ct_1}")
            self.get_logger().info(f"  Moment coefficients - cm_0: {self.cm_0}, cm_1: {self.cm_1}")
            self.get_logger().info(f"  PWM conversion - p1: {self.pwm_p1}, p2: {self.pwm_p2}, p3: {self.pwm_p3}")
            self.get_logger().info(f"  Control allocation matrix B:\n{self.B}")
            self.get_logger().info(f"  Control allocation inverse B_inv:\n{self.B_inv}")
    
    def create_control_allocation_matrices(self) -> None:
        """
        Create control allocation matrices using loaded parameters.
        
        This method creates the B matrix for control allocation mapping from
        desired forces/moments to motor thrust/moment commands, and computes
        its inverse for the control allocation algorithm.
        
        The B matrix maps [thrust_motor_0, thrust_motor_1] to [moment_y, thrust_z]
        based on the thrust and moment coefficients loaded from parameters.
        """
        # Create control allocation matrix B
        # B maps motor commands to [moment_y, thrust_z]
        self.B = np.array([
            [self.cm_0 * self.ct_0, -self.cm_1 * self.ct_1],  # moment_y row
            [self.ct_0, self.ct_1]                            # thrust_z row
        ])
        
        # Compute pseudo-inverse for control allocation
        try:
            self.B_inv = np.linalg.inv(self.B)
        except np.linalg.LinAlgError:
            self.get_logger().error("Control allocation matrix B is singular! Using pseudo-inverse.")
            self.B_inv = np.linalg.pinv(self.B)

    
    def get_physical_properties(self) -> Dict[str, float]:
        """
        Return physical properties as a dictionary.
        
        Returns:
            Dict[str, float]: Dictionary containing vehicle physical properties
                including gravity (G), center of mass to thrust distance,
                moments of inertia (I_XX, I_YY, I_ZZ), and mass.
        """
        return {
            'G': self.G,
            'DIST_COM_2_THRUST': self.DIST_COM_2_THRUST,
            'I_XX': self.I_XX,
            'I_YY': self.I_YY,
            'I_ZZ': self.I_ZZ,
            'MASS': self.MASS
        }
    
    def get_lqr_matrices(self) -> tuple:
        """
        Return Q and R matrices for LQR controller.
        
        Returns:
            tuple: (Q_matrix, R_matrix) where Q_matrix is the state weighting
                   matrix and R_matrix is the control input weighting matrix
                   for the LQR controller.
        """
        return self.Q_matrix, self.R_matrix
    
    def get_target_state(self) -> np.ndarray:
        """
        Return target state vector for LQR controller.
        
        Returns:
            np.ndarray: Copy of the target state vector containing desired
                       position, velocity, quaternion, and angular velocity.
        """
        return self.target_state.copy()
    
    def get_inertia_matrix(self) -> np.ndarray:
        """
        Return inertia matrix.
        
        Returns:
            np.ndarray: Copy of the 3x3 inertia matrix with diagonal elements
                       [I_XX, I_YY, I_ZZ].
        """
        return self.inertia_matrix.copy()
    
    def update_target_position(self, x: float, y: float, z: float):
        """
        Update target position and recreate target state vector.
        
        Args:
            x (float): Target x-position in NED frame [meters]
            y (float): Target y-position in NED frame [meters]  
            z (float): Target z-position in NED frame [meters]
        """
        self.target_position = np.array([x, y, z])
        self.target_state = np.concatenate([
            self.target_position,
            self.target_velocity,
            self.target_quaternion,  
            self.target_angular_velocity
        ])
        self.get_logger().info(f"Updated target position to: {self.target_position}")

    def get_control_allocation_parameters(self) -> Dict[str, Any]:
        """
        Return control allocation parameters as a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary containing all control allocation parameters
                        including coefficients, matrices, and configuration flags.
        """
        return {
            'use_onboard_computer': self.use_onboard_control_allocation,
            'thrust_coefficients': {
                'ct_0': self.ct_0,
                'ct_1': self.ct_1
            },
            'moment_coefficients': {
                'cm_0': self.cm_0,
                'cm_1': self.cm_1
            },
            'pwm_conversion': {
                'p1': self.pwm_p1,
                'p2': self.pwm_p2,
                'p3': self.pwm_p3
            },
            'allocation_matrix': self.B.copy(),
            'allocation_inverse': self.B_inv.copy()
        }
    
    def vehicle_attitude_gt_callback(self, msg: VehicleAttitude):
        """
        Callback for vehicle attitude ground truth messages.
        
        This callback processes incoming attitude data from PX4, handles
        quaternion format conversion from [qw, qx, qy, qz] to [qx, qy, qz, qw],
        applies orientation normalization if initialized, and manages the
        attitude initialization phase including data collection for averaging.
        
        Args:
            msg (VehicleAttitude): ROS2 message containing vehicle attitude
                                 information including quaternion orientation.
        """
        # Extract quaternion [qw, qx, qy, qz] from msg
        q_msg = msg.q 
        
        # Handle invalid quaternion
        if np.isnan(q_msg[0]):
            if self._last_update_quaternion is not None:
                self.current_quaternion = self._last_update_quaternion.copy()
            else:
                self.current_quaternion = np.array([0, 0, 0, 1])  # Default quaternion
            return
        
        # Reorder to [qx, qy, qz, qw] for internal use
        current_q_ned = np.array([q_msg[1], q_msg[2], q_msg[3], q_msg[0]])
        
        # Store valid quaternion
        self.current_quaternion = current_q_ned
        self._last_update_quaternion = current_q_ned.copy()
        
        # Apply orientation normalization if initialized
        if self.is_orientation_normalized:
            self._apply_orientation_normalization()
            # self._last_update_quaternion = self.current_quaternion.copy()
        
        # Handle initialization phase
        self._handle_attitude_initialization()
        
    def groundtruth_odometry_callback(self, msg: Odometry):
        # Extract state information
        self.current_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.current_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.current_angular_velocity = np.array([msg.twist.twist.angular.x, -msg.twist.twist.angular.y, -msg.twist.twist.angular.z])
        
        # Check for NaN values
        if (np.any(np.isnan(self.current_position)) or 
            np.any(np.isnan(self.current_velocity)) or 
            np.any(np.isnan(self.current_angular_velocity))):
            self.get_logger().warn("Vehicle odometry contains NaN values - skipping update")
            return
        
        # Store valid data
        self._last_update_position = self.current_position.copy()
        self._last_update_velocity = self.current_velocity.copy()
        self._last_update_angular_velocity = self.current_angular_velocity.copy()
        
        # Initialize odometry state flag
        if not self.is_odom_state_initialized:
            self.is_odom_state_initialized = True
            self.get_logger().info("Vehicle odometry state initialized")
        
        # Handle initialization phase
        self._handle_odometry_initialization()
        
        # q_msg = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z])
        
        # # Handle invalid quaternion
        # if np.isnan(q_msg[0]):
        #     if self._last_update_quaternion is not None:
        #         self.current_quaternion = self._last_update_quaternion.copy()
        #     else:
        #         self.current_quaternion = np.array([0, 0, 0, 1])  # Default quaternion
        #     return
        
        # # Reorder to [qx, qy, qz, qw] for internal use
        # current_q_ned = np.array([q_msg[1], q_msg[2], q_msg[3], q_msg[0]])
        
        # # Store valid quaternion
        # self.current_quaternion = current_q_ned
        # self._last_update_quaternion = current_q_ned.copy()
        
        # # Apply orientation normalization if initialized
        # if self.is_orientation_normalized:
        #     self._apply_orientation_normalization()
        #     # self._last_update_quaternion = self.current_quaternion.copy()
        
        # # Handle initialization phase
        # self._handle_attitude_initialization()

    def vehicle_odometry_callback(self, msg: VehicleOdometry):
        """
        Callback for vehicle odometry messages.
        
        This callback processes incoming odometry data from PX4, extracts
        position, velocity, and angular velocity information, performs
        NaN value checking for data validity, and manages the odometry
        initialization phase including data collection for position averaging.
        
        Args:
            msg (VehicleOdometry): ROS2 message containing vehicle odometry
                                 information including position, velocity,
                                 and angular velocity in NED/FRD frames.
        """
        # Extract state information
        self.current_position = np.array(msg.position)
        self.current_velocity = np.array(msg.velocity)
        self.current_angular_velocity = np.array(msg.angular_velocity)
        
        # Check for NaN values
        if (np.any(np.isnan(self.current_position)) or 
            np.any(np.isnan(self.current_velocity)) or 
            np.any(np.isnan(self.current_angular_velocity))):
            self.get_logger().warn("Vehicle odometry contains NaN values - skipping update")
            return
        
        # Store valid data
        self._last_update_position = self.current_position.copy()
        self._last_update_velocity = self.current_velocity.copy()
        self._last_update_angular_velocity = self.current_angular_velocity.copy()
        
        # Initialize odometry state flag
        if not self.is_odom_state_initialized:
            self.is_odom_state_initialized = True
            self.get_logger().info("Vehicle odometry state initialized")
        
        # Handle initialization phase
        self._handle_odometry_initialization()
        
        # Extract quaternion [qw, qx, qy, qz] from msg
        q_msg = msg.q 
        
        # Handle invalid quaternion
        if np.isnan(q_msg[0]):
            if self._last_update_quaternion is not None:
                self.current_quaternion = self._last_update_quaternion.copy()
            else:
                self.current_quaternion = np.array([0, 0, 0, 1])  # Default quaternion
            return
        
        # Reorder to [qx, qy, qz, qw] for internal use
        current_q_ned = np.array([q_msg[1], q_msg[2], q_msg[3], q_msg[0]])
        
        # Store valid quaternion
        self.current_quaternion = current_q_ned
        self._last_update_quaternion = current_q_ned.copy()
        
        # Apply orientation normalization if initialized
        if self.is_orientation_normalized:
            self._apply_orientation_normalization()
            # self._last_update_quaternion = self.current_quaternion.copy()
        
        # Handle initialization phase
        self._handle_attitude_initialization()

    # Helper methods to keep callbacks clean

    def _apply_orientation_normalization(self):
        """
        Apply the fixed rotation matrix to normalize orientation.
        
        This method transforms the current quaternion using the initial
        rotation matrix computed during initialization to normalize the
        orientation reference frame. It ensures the quaternion scalar
        part remains positive for consistency.
        """
        r_current = R.from_quat(self.current_quaternion)
        r_normalized = R.from_matrix(self.initial_rotation_matrix) * r_current
        self.current_quaternion = r_normalized.as_quat()
        
        # Ensure positive scalar part
        if self.current_quaternion[3] < 0:
            self.current_quaternion = -self.current_quaternion

    def _handle_attitude_initialization(self):
        """
        Handle attitude initialization phase with simplified logic.
        
        This method manages the collection of attitude samples during the
        first 100 callback invocations, then initializes the orientation
        system once sufficient data has been collected. This provides
        a stable reference for attitude control.
        """
        if not self.is_attitude_gt_initialized:
            self.quaternion_gt_callback_counter += 1
            
            # Collect samples for averaging (first 100 readings)
            if self.quaternion_gt_callback_counter <= 100:
                if self._last_update_quaternion is not None:
                    self._list_of_quaternions.append(self._last_update_quaternion.copy())
            
            # Initialize after collecting samples
            elif self.quaternion_gt_callback_counter == 101:
                self._initialize_orientation_system()
                self.is_attitude_gt_initialized = True

    def _handle_odometry_initialization(self):
        """
        Handle odometry initialization phase with simplified logic.
        
        This method manages the collection of position samples during the
        first 100 callback invocations, then initializes the position
        targets once sufficient data has been collected. This establishes
        a stable reference for position control.
        """
        if not self.is_final_position_state_initialized:
            self.odom_state_callback_counter += 1
            
            # Collect samples for averaging (first 100 readings)
            if self.odom_state_callback_counter <= 100:
                self._list_of_positions.append(self._last_update_position.copy())
            
            # Initialize after collecting samples
            elif self.odom_state_callback_counter == 101:
                self._initialize_position_targets()
                self.is_final_position_state_initialized = True

    def _initialize_orientation_system(self):
        """
        Initialize orientation normalization system.
        
        This method computes the average quaternion from collected samples,
        creates the initial rotation matrix for orientation normalization,
        and sets the quaternion target to [0,0,0,1] in the normalized frame.
        This provides a consistent orientation reference regardless of the
        initial vehicle attitude.
        """
        if len(self._list_of_quaternions) == 0:
            self.get_logger().error("No quaternion data collected for initialization!")
            return
        
        # Calculate average quaternion and normalize
        quaternions_array = np.array(self._list_of_quaternions)
        avg_quaternion = np.mean(quaternions_array, axis=0)
        avg_quaternion = avg_quaternion / np.linalg.norm(avg_quaternion)
        
        self.get_logger().info(f"Average quaternion: {avg_quaternion}")
        
        # Initialize normalization matrix
        r_avg = R.from_quat(avg_quaternion)
        self.initial_rotation_matrix = r_avg.inv().as_matrix()
        self.is_orientation_normalized = True
        
        # Set quaternion target (always [0,0,0,1] in normalized frame)
        self.quaternion_target = self.target_quaternion.copy()
        
        self.get_logger().info("Orientation normalization initialized")

    def _initialize_position_targets(self):
        """
        Initialize position targets based on collected data.
        
        This method computes the average position from collected samples
        and sets the final position targets by adding the configured
        target offsets from YAML parameters. This establishes the
        desired hover position relative to the initial vehicle location.
        """
        if len(self._list_of_positions) == 0:
            self.get_logger().error("No position data collected for initialization!")
            return
        
        # Calculate average position
        positions_array = np.array(self._list_of_positions)
        avg_position = np.mean(positions_array, axis=0)
        
        # Set targets using YAML parameters instead of hardcoded X_F_TARGET
        self.position_target = avg_position + self.target_position
        self.velocity_target = self.target_velocity.copy()
        self.angular_velocity_target = self.target_angular_velocity.copy()
        
        self.get_logger().info(f"Target position: {self.position_target}")
        self.get_logger().info("Position targets initialized")

    def _check_controller_ready(self):
        """
        Check if controller is ready to start.
        
        Returns:
            bool: True if all required initializations are complete
                  (attitude, position, and odometry), False otherwise.
        """
        return (self.is_attitude_gt_initialized and 
                self.is_final_position_state_initialized and
                self.is_odom_state_initialized)

    def vehicle_status_callback(self, msg: VehicleStatus):
        """
        Callback for vehicle status messages.
        
        This callback updates the vehicle's navigation state and arming state
        from PX4 status messages, which are used to monitor vehicle mode
        and safety status during operation.
        
        Args:
            msg (VehicleStatus): ROS2 message containing vehicle status
                               information including navigation and arming states.
        """
        self.vehicle_status = msg
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state
    
    def arm(self):
        """
        Send an arm command to the vehicle.
        
        This method publishes a vehicle command to arm the motors, enabling
        the vehicle for flight operations. The command is sent with param1=1.0
        to indicate arming request.
        """
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')
    
    def disarm(self):
        """
        Send a disarm command to the vehicle.
        
        This method publishes a vehicle command to disarm the motors, disabling
        the vehicle for safety. The command is sent with param1=0.0 to indicate
        disarming request.
        """
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')
    
    def engage_offboard_mode(self):
        """
        Switch to offboard mode.
        
        This method publishes a vehicle command to switch PX4 into offboard
        control mode, allowing external control of the vehicle through ROS2
        messages. Uses mode parameters param1=1.0 (custom mode) and param2=6.0
        (offboard mode).
        """
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Switching to offboard mode')
    
    def publish_offboard_control_heartbeat_signal(self):
        """
        Publish the offboard control mode heartbeat signal.
        
        This method publishes an OffboardControlMode message to maintain
        offboard control authority with PX4. It specifies that thrust_and_torque
        control is enabled while disabling other control modes. This heartbeat
        must be published continuously to maintain offboard control.
        """
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        if self.use_onboard_control_allocation:
            msg.thrust_and_torque = False 
            msg.direct_actuator = True
        else:   
            msg.thrust_and_torque = True 
            msg.direct_actuator = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)
    
    def publish_thrust_setpoint(self, thrust_sp=-0.0):
        """
        Publish thrust setpoint in body frame.
        
        This method publishes a VehicleThrustSetpoint message containing the
        desired thrust vector in the vehicle body frame. The thrust is applied
        along the z-axis (downward positive in FRD frame).
        
        Args:
            thrust_sp (float, optional): Thrust setpoint in Newtons along
                                       body z-axis. Defaults to -0.0.
        """
        msg = VehicleThrustSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.xyz = [0.0, 0.0, thrust_sp]  # Thrust in body frame [N]
        self.thrust_setpoint_publisher.publish(msg)
    
    def publish_torque_setpoint(self, phi, theta, torque_sp_yaw=0.0):
        """
        Publish torque setpoint in body frame.
        
        This method publishes a VehicleTorqueSetpoint message containing the
        desired torque vector in the vehicle body frame for thrust vector
        control. The phi and theta parameters control gimbal deflection angles.
        
        Args:
            phi (float): Roll torque/gimbal deflection angle [radians or Nm]
            theta (float): Pitch torque/gimbal deflection angle [radians or Nm] 
            torque_sp_yaw (float, optional): Yaw torque setpoint [Nm].
                                            Defaults to 0.0.
        """
        msg = VehicleTorqueSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.xyz = [phi, theta, float(torque_sp_yaw)]  # Torque in body frame [Nm]
        self.torque_setpoint_publisher.publish(msg)

    def publish_actuator_motor_setpoints(self, pwm_commands=None):
        """
        Publish actuator motor setpoints for thrust vector control.
        
        This method publishes normalized PWM commands to the motor actuators.
        The PWM commands are calculated from the control allocation process
        and represent the desired motor speeds for thrust generation.
        
        Args:
            pwm_commands (np.ndarray, optional): Normalized PWM commands [0,1]
                                               for motors. Defaults to zeros if None.
        """
        if pwm_commands is None:
            # Default to zero if no commands provided
            pwm_commands = np.zeros((2, 1))
        
        # Ensure commands are in [0, 1] range
        pwm_commands = np.clip(pwm_commands, 0.0, 1.0)  

        motor_data = ActuatorMotors()
        motor_data.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        motor_data.control = [pwm_commands[0][0], pwm_commands[1][0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        motor_data.reversible_flags = False
        self.actuator_motor_setpoint_publisher.publish(motor_data)

    def publish_actuator_servo_setpoints(self, phi=None, theta=None):
        """
        Publish actuator servo setpoints for gimbal control.
        
        This method publishes servo angle commands for thrust vector control
        gimbal actuation. The servo angles control the orientation of the
        thrust vector to provide attitude control.
        
        Args:
            phi (float, optional): Roll servo angle [radians]. Defaults to 0.0.
            theta (float, optional): Pitch servo angle [radians]. Defaults to 0.0.
        """

        if phi is None:
            # Default to zero if no phi provided
            phi = 0.0
        if theta is None:
            # Default to zero if no theta provided
            theta = 0.0

        phi = np.clip(phi, -1.0, 1.0)  # Limit servo angles to [-1, 1] range
        theta = np.clip(theta, -1.0, 1.0)  # Limit servo angles to [-1, 1] range
        
        servo_data = ActuatorServos()
        servo_data.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        servo_data.control = [phi, theta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.actuator_servo_setpoint_publisher.publish(servo_data)
    
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param3=0.0):
        """
        Publish a vehicle command to PX4.
        
        This method creates and publishes a VehicleCommand message to send
        commands to the PX4 autopilot such as arming, mode changes, or other
        vehicle-level commands.
        
        Args:
            command (int): Vehicle command ID from VehicleCommand constants
            param1 (float, optional): First command parameter. Defaults to 0.0.
            param2 (float, optional): Second command parameter. Defaults to 0.0.
            param3 (float, optional): Third command parameter. Defaults to 0.0.
        """
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.param3 = param3
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)
        return
    
    def _cal_quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions using Hamilton product.
        
        This method performs quaternion multiplication following the Hamilton
        convention. Both input quaternions should be in [qx, qy, qz, qw] format.
        
        Args:
            q1 (np.ndarray): First quaternion [qx, qy, qz, qw]
            q2 (np.ndarray): Second quaternion [qx, qy, qz, qw]
            
        Returns:
            np.ndarray: Result of q1 * q2 in [qx, qy, qz, qw] format
        """
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([x, y, z, w])
    
    def _cal_quaternion_inverse(self, q):
        """
        Calculate the inverse (conjugate) of a unit quaternion.
        
        For a unit quaternion, the inverse is equal to its conjugate,
        which negates the vector part while keeping the scalar part unchanged.
        
        Args:
            q (np.ndarray): Input quaternion [qx, qy, qz, qw]
            
        Returns:
            np.ndarray: Quaternion inverse [-qx, -qy, -qz, qw]
        """
        return np.array([-q[0], -q[1], -q[2], q[3]])
    
    def _cal_quaternion_error(self, q_current, q_target):
        """
        Calculate quaternion error between current and target orientations.
        
        This method computes the quaternion that represents the rotation
        needed to go from the target orientation to the current orientation.
        The error quaternion is calculated as q_current * q_target^(-1).
        
        Args:
            q_current (np.ndarray): Current quaternion [qx, qy, qz, qw]
            q_target (np.ndarray): Target quaternion [qx, qy, qz, qw]
            
        Returns:
            np.ndarray: Error quaternion [qx, qy, qz, qw]
        """
        q_diff = self._cal_quaternion_multiply(
            q_current, self._cal_quaternion_inverse(q_target))
        return q_diff
    
    def timer_callback(self):
        """
        Main timer callback for vehicle sequencing at 10Hz.
        
        This callback manages the startup sequence for the vehicle:
        - At count 10: Send arm command
        - At count 30: Engage offboard mode
        - Continues counting up to 5000 for sequence timing
        
        This provides a controlled startup sequence with appropriate delays
        between arming and mode switching operations.
        """
        
        if self.offboard_setpoint_counter == 10:
            self.arm()

        if self.offboard_setpoint_counter == 30:
            self.engage_offboard_mode()
        
        if self.offboard_setpoint_counter < 5000:
            self.offboard_setpoint_counter += 1
  
    def allocate(self, phi, theta, thrust, torque):
        """
        Allocate control inputs to motor and servo actuators.
        
        This method performs control allocation by mapping the desired control
        inputs (phi, theta, thrust, torque) to individual actuator commands.
        For onboard control allocation, it converts desired moments and thrust
        to motor speeds using the control allocation matrix, then converts
        motor speeds to PWM commands using polynomial coefficients.
        
        Args:
            phi (float): Roll torque/gimbal deflection angle [radians or Nm]
            theta (float): Pitch torque/gimbal deflection angle [radians or Nm]
            thrust (float): Thrust setpoint in gimbal z-frame [N]
            torque (float): Torque setpoint in gimbal z-frame [Nm]
        """
        thrust *= -1
        
        control_input = np.array([
            [torque],
            [thrust]])
        
        motor_speeds_square = self.B_inv @ control_input
        
        # If motor_speeds_square is negative, set to nan to avoid sqrt of negative else sqrt
        if np.any(motor_speeds_square < 0):
            self.get_logger().warn("Negative motor speeds detected, setting to NaN")
            pwm_commands_normalized = np.array([[np.NaN], [np.NaN]])
        else:
            motor_speeds = np.sqrt(motor_speeds_square)
            
            # Calculate PWM commands using the polynomial coefficients
            # PWM = p1* motor_speeds^2 + p2 * motor_speeds + p3
            pwm_commands = self.pwm_p1 * motor_speeds**2 +\
                            self.pwm_p2 * motor_speeds +\
                            self.pwm_p3

            # Clip PWM commands to valid range
            pwm_commands = np.clip(pwm_commands, 1100, 1900)
            
            # Normalize PWM commands to [0, 1] range
            pwm_commands_normalized = (pwm_commands - 1000) / 1000

        # self.get_logger().info(f'PWM Commands Normalized: {pwm_commands_normalized}, phi: {phi}, theta: {theta}')
        print("pwm_commands_normalized", pwm_commands_normalized)
        self.publish_actuator_motor_setpoints(pwm_commands_normalized)
        self.publish_actuator_servo_setpoints(phi, theta)   

    
    def control_timer(self):
        """
        High-frequency control timer callback at 100Hz.
        
        This is the main control loop that:
        1. Publishes offboard control heartbeat to maintain control authority
        2. Waits for all systems to be initialized and ready
        3. Calculates state errors (position, velocity, attitude, angular velocity)
        4. Applies LQR control law to compute control inputs
        5. Applies actuator constraints and publishes thrust/torque setpoints
        
        The control loop implements the complete TVC control system using
        Linear Quadratic Regulator (LQR) feedback control.
        """
        self.publish_offboard_control_heartbeat_signal()
        
        # Only run controller when everything is ready
        if not self._check_controller_ready():
            return
        
        # Calculate state errors
        pos_error = self._last_update_position - self.position_target
        vel_error = self._last_update_velocity - self.velocity_target
        
        # limit the velocity error to 0.5 m/s
        # vel_error = np.clip(vel_error, -0.2, 0.2)
        
        quaternion_error = self._cal_quaternion_error(self._last_update_quaternion, self.quaternion_target)
        angular_velocity_error = self._last_update_angular_velocity - self.angular_velocity_target
        
        # Combine errors into state vector
        x_error = np.concatenate([
            pos_error,
            vel_error, 
            quaternion_error[:3],  # Only xyz components
            angular_velocity_error
        ])
        
        self.get_logger().info(f'x_error: {x_error}')
        
        # Apply LQR control
        u_lqr = -self.K @ x_error
        self.get_logger().info(f'LQR Control Output: {u_lqr}')
        
        # Extract and limit control inputs using YAML parameters
        phi = np.clip(u_lqr[0], self.servo_0_min, self.servo_0_max)
        theta = np.clip(u_lqr[1], self.servo_1_min, self.servo_1_max)
        thrust_without_gravity = u_lqr[2] - self.MASS * self.G
        thrust_gimbal_z_frame = np.clip(
            thrust_without_gravity, 
            self.thrust_min_g_multiplier * self.G, 
            self.MASS * self.G * self.thrust_max_weight_fraction
        )
        torque_gimbal_z_frame = np.clip(u_lqr[3], self.torque_z_min, self.torque_z_max)
        
        self.get_logger().info(f'phi: {phi}, theta: {theta}, thrust_gimbal_z_frame: {thrust_gimbal_z_frame}, torque_gimbal_z_frame: {torque_gimbal_z_frame}')
        
        # Publish control commands using the selected method
        if self.use_onboard_control_allocation:
            self.allocate(phi, theta, thrust_gimbal_z_frame, torque_gimbal_z_frame)
        else:   
            self.publish_thrust_setpoint(thrust_gimbal_z_frame)
            self.publish_torque_setpoint(phi, theta, torque_gimbal_z_frame)
        

def main(args=None) -> None:
    """
    Main function to initialize and run the TVC controller.
    
    This function initializes the ROS2 system, creates the TVC controller
    node, and enters the ROS2 spin loop to process callbacks. It handles
    proper cleanup and shutdown when the program exits.
    
    Args:
        args: Command line arguments passed to ROS2 initialization
    """
    print('Starting TVC Controller...')
    rclpy.init(args=args)
    px4_controller = TVC_CONTROLLER()
    rclpy.spin(px4_controller)
    px4_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)