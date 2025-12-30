#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
import math


class AutoTakeoffLand(Node):
    """Node for automatic takeoff, translation, and landing."""

    # State machine states
    STATE_INIT = 0
    STATE_ARMING = 1
    STATE_TAKEOFF = 2
    STATE_TRANSLATING = 3
    STATE_LANDING = 4
    STATE_DISARMING = 5

    def __init__(self) -> None:
        super().__init__('auto_takeoff_land')

        # Configure QoS profile for publishing and subscribing
        # Using depth=5 to accommodate larger message payloads and avoid RTPS warnings
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        
        # Mission parameters
        self.takeoff_height = -1.0  # 1m height in NED frame (negative is up)
        self.translation_distance = 1.0  # 1m along X axis
        self.position_tolerance = 0.1  # Position tolerance in meters
        self.altitude_tolerance = 0.1  # Altitude tolerance in meters
        
        # State machine
        self.state = self.STATE_INIT
        self.initial_position = None
        self.target_position = [0.0, 0.0, 0.0]
        
        # Create a timer to publish control commands at 10Hz
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info('Auto takeoff and land node initialized')

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position
        
        # Store initial position on first valid reading
        if (self.initial_position is None and 
            vehicle_local_position.xy_valid and 
            vehicle_local_position.z_valid):
            self.initial_position = [
                vehicle_local_position.x,
                vehicle_local_position.y,
                vehicle_local_position.z
            ]
            self.get_logger().info(f'Initial position recorded: {self.initial_position}')

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode heartbeat."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float, yaw: float = 0.0):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoint: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def is_position_reached(self, target_pos, current_pos, tolerance=None):
        """Check if current position is within tolerance of target position."""
        if tolerance is None:
            tolerance = self.position_tolerance
        
        if not (self.vehicle_local_position.xy_valid and self.vehicle_local_position.z_valid):
            return False
        
        dx = current_pos[0] - target_pos[0]
        dy = current_pos[1] - target_pos[1]
        dz = current_pos[2] - target_pos[2]
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        return distance < tolerance

    def is_altitude_reached(self, target_z, current_z, tolerance=None):
        """Check if current altitude is within tolerance of target altitude."""
        if tolerance is None:
            tolerance = self.altitude_tolerance
        
        if not self.vehicle_local_position.z_valid:
            return False
        
        dz = abs(current_z - target_z)
        return dz < tolerance

    def timer_callback(self) -> None:
        """Main timer callback implementing state machine."""
        # Always publish offboard control heartbeat
        self.publish_offboard_control_heartbeat_signal()

        # Get current position
        if (self.vehicle_local_position.xy_valid and 
            self.vehicle_local_position.z_valid):
            current_pos = [
                self.vehicle_local_position.x,
                self.vehicle_local_position.y,
                self.vehicle_local_position.z
            ]
        else:
            current_pos = None

        # State machine logic
        if self.state == self.STATE_INIT:
            # Send offboard mode and arm commands after sending some setpoints
            if self.offboard_setpoint_counter == 10:
                self.engage_offboard_mode()
                self.arm()
            
            if self.offboard_setpoint_counter < 11:
                self.offboard_setpoint_counter += 1
            else:
                # Wait for arming and offboard mode
                if (self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED and
                    self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
                    self.state = self.STATE_TAKEOFF
                    if self.initial_position is not None:
                        self.target_position = [
                            self.initial_position[0],
                            self.initial_position[1],
                            self.initial_position[2] + self.takeoff_height
                        ]
                    else:
                        # Fallback if initial position not available
                        self.target_position = [0.0, 0.0, self.takeoff_height]
                    self.get_logger().info(f'Transitioning to TAKEOFF state. Target: {self.target_position}')

        elif self.state == self.STATE_TAKEOFF:
            # Publish takeoff position setpoint
            self.publish_position_setpoint(
                self.target_position[0],
                self.target_position[1],
                self.target_position[2]
            )
            
            # Check if altitude reached
            if current_pos is not None:
                if self.is_altitude_reached(self.target_position[2], current_pos[2]):
                    self.state = self.STATE_TRANSLATING
                    # Set new target position: translate 1m along X axis
                    if self.initial_position is not None:
                        self.target_position = [
                            self.initial_position[0] + self.translation_distance,
                            self.initial_position[1],
                            self.initial_position[2] + self.takeoff_height
                        ]
                    else:
                        self.target_position = [
                            self.translation_distance,
                            0.0,
                            self.takeoff_height
                        ]
                    self.get_logger().info(f'Takeoff complete. Transitioning to TRANSLATING state. Target: {self.target_position}')

        elif self.state == self.STATE_TRANSLATING:
            # Publish translation position setpoint
            self.publish_position_setpoint(
                self.target_position[0],
                self.target_position[1],
                self.target_position[2]
            )
            
            # Check if position reached
            if current_pos is not None:
                if self.is_position_reached(self.target_position, current_pos):
                    self.state = self.STATE_LANDING
                    self.get_logger().info('Translation complete. Transitioning to LANDING state.')
                    self.land()

        elif self.state == self.STATE_LANDING:
            # Landing command already sent, just wait
            # Check if vehicle has landed (z position close to initial or positive)
            if current_pos is not None and self.initial_position is not None:
                # Check if close to ground (within tolerance of initial altitude)
                if abs(current_pos[2] - self.initial_position[2]) < self.altitude_tolerance:
                    self.state = self.STATE_DISARMING
                    self.get_logger().info('Landing complete. Disarming vehicle.')
                    self.disarm()
            elif current_pos is not None:
                # Fallback: if z is positive (above ground in NED), consider landed
                if current_pos[2] > -0.2:  # Close to ground
                    self.state = self.STATE_DISARMING
                    self.get_logger().info('Landing complete. Disarming vehicle.')
                    self.disarm()

        elif self.state == self.STATE_DISARMING:
            # Wait for disarming, then shutdown
            if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.get_logger().info('Vehicle disarmed. Mission complete. Shutting down.')
                rclpy.shutdown()


def main(args=None) -> None:
    print('Starting auto takeoff and land node...')
    rclpy.init(args=args)
    auto_takeoff_land = AutoTakeoffLand()
    rclpy.spin(auto_takeoff_land)
    auto_takeoff_land.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')

