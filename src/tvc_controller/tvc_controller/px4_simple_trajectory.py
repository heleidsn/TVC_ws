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
    STATE_HOVERING = 4
    STATE_LANDING = 5
    STATE_DISARMING = 6

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
            VehicleStatus, '/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.last_nav_state = None
        self.last_arming_state = None
        
        # Mission parameters
        self.takeoff_height = -1.5  # 1m height in NED frame (negative is up)
        self.translation_distance = 2.0  # 1m along X axis
        self.position_tolerance = 0.05  # Position tolerance in meters
        self.altitude_tolerance = 0.05  # Altitude tolerance in meters
        self.hover_duration = 2.0  # Hover duration in seconds
        
        # State machine
        self.state = self.STATE_INIT
        self.initial_position = None
        self.target_position = [0.0, 0.0, 0.0]
        self.hover_start_time = None
        
        # Status display counter (display status every 10 timer cycles = 1 second)
        self.status_display_counter = 0
        self.status_display_interval = 10
        
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
        # print(f'vehicle_status: {vehicle_status}')
        # Log status changes for debugging
        if (hasattr(self, 'last_nav_state') and 
            self.last_nav_state != vehicle_status.nav_state):
            self.get_logger().info(
                f'Nav state changed: {self.last_nav_state} -> {vehicle_status.nav_state}'
            )
        if (hasattr(self, 'last_arming_state') and 
            self.last_arming_state != vehicle_status.arming_state):
            self.get_logger().info(
                f'Arming state changed: {self.last_arming_state} -> {vehicle_status.arming_state}'
            )
        self.last_nav_state = vehicle_status.nav_state
        self.last_arming_state = vehicle_status.arming_state

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
        # self.get_logger().info(f"Publishing position setpoint: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")

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

    def get_state_name(self, state):
        """Get human-readable state name."""
        state_names = {
            self.STATE_INIT: "INIT",
            self.STATE_ARMING: "ARMING",
            self.STATE_TAKEOFF: "TAKEOFF",
            self.STATE_TRANSLATING: "TRANSLATING",
            self.STATE_HOVERING: "HOVERING",
            self.STATE_LANDING: "LANDING",
            self.STATE_DISARMING: "DISARMING"
        }
        return state_names.get(state, "UNKNOWN")

    def get_nav_state_name(self, nav_state):
        """Get human-readable navigation state name."""
        nav_state_names = {
            VehicleStatus.NAVIGATION_STATE_MANUAL: "MANUAL",
            VehicleStatus.NAVIGATION_STATE_ALTCTL: "ALTCTL",
            VehicleStatus.NAVIGATION_STATE_POSCTL: "POSCTL",
            VehicleStatus.NAVIGATION_STATE_AUTO_MISSION: "AUTO_MISSION",
            VehicleStatus.NAVIGATION_STATE_AUTO_LOITER: "AUTO_LOITER",
            VehicleStatus.NAVIGATION_STATE_AUTO_RTL: "AUTO_RTL",
            VehicleStatus.NAVIGATION_STATE_POSITION_SLOW: "POSITION_SLOW",
            VehicleStatus.NAVIGATION_STATE_FREE5: "FREE5",
            VehicleStatus.NAVIGATION_STATE_FREE4: "FREE4",
            VehicleStatus.NAVIGATION_STATE_FREE3: "FREE3",
            VehicleStatus.NAVIGATION_STATE_ACRO: "ACRO",
            VehicleStatus.NAVIGATION_STATE_FREE2: "FREE2",
            VehicleStatus.NAVIGATION_STATE_DESCEND: "DESCEND",
            VehicleStatus.NAVIGATION_STATE_TERMINATION: "TERMINATION",
            VehicleStatus.NAVIGATION_STATE_OFFBOARD: "OFFBOARD",
            VehicleStatus.NAVIGATION_STATE_STAB: "STAB",
            VehicleStatus.NAVIGATION_STATE_FREE1: "FREE1",
            VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF: "AUTO_TAKEOFF",
            VehicleStatus.NAVIGATION_STATE_AUTO_LAND: "AUTO_LAND",
            VehicleStatus.NAVIGATION_STATE_AUTO_FOLLOW_TARGET: "AUTO_FOLLOW_TARGET",
            VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND: "AUTO_PRECLAND",
            VehicleStatus.NAVIGATION_STATE_ORBIT: "ORBIT",
            VehicleStatus.NAVIGATION_STATE_AUTO_VTOL_TAKEOFF: "AUTO_VTOL_TAKEOFF",
            VehicleStatus.NAVIGATION_STATE_EXTERNAL1: "EXTERNAL1",
            VehicleStatus.NAVIGATION_STATE_EXTERNAL2: "EXTERNAL2",
            VehicleStatus.NAVIGATION_STATE_EXTERNAL3: "EXTERNAL3",
            VehicleStatus.NAVIGATION_STATE_EXTERNAL4: "EXTERNAL4",
            VehicleStatus.NAVIGATION_STATE_EXTERNAL5: "EXTERNAL5",
            VehicleStatus.NAVIGATION_STATE_EXTERNAL6: "EXTERNAL6",
            VehicleStatus.NAVIGATION_STATE_EXTERNAL7: "EXTERNAL7",
            VehicleStatus.NAVIGATION_STATE_EXTERNAL8: "EXTERNAL8",
            VehicleStatus.NAVIGATION_STATE_MAX: "MAX"
        }
        return nav_state_names.get(nav_state, f"UNKNOWN({nav_state})")

    def get_arming_state_name(self, arming_state):
        """Get human-readable arming state name."""
        arming_state_names = {
            0: "INIT/UNKNOWN",
            1: "DISARMED",
            2: "ARMED",
            VehicleStatus.ARMING_STATE_DISARMED: "DISARMED",
            VehicleStatus.ARMING_STATE_ARMED: "ARMED"
        }
        return arming_state_names.get(arming_state, f"UNKNOWN({arming_state})")

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

        # Display status periodically (every 1 second)
        self.status_display_counter += 1
        if self.status_display_counter >= self.status_display_interval:
            self.status_display_counter = 0
            
            # Display status information directly in timer_callback
            state_name = self.get_state_name(self.state)
            nav_state_name = self.get_nav_state_name(self.vehicle_status.nav_state)
            arming_state_name = self.get_arming_state_name(self.vehicle_status.arming_state)
            
            # Position information
            if current_pos is not None:
                pos_str = f"Pos: x={current_pos[0]:.3f}, y={current_pos[1]:.3f}, z={current_pos[2]:.3f}"
                pos_valid = "✓" if (self.vehicle_local_position.xy_valid and self.vehicle_local_position.z_valid) else "✗"
            else:
                pos_str = "Pos: INVALID"
                pos_valid = "✗"
            
            # Target position
            target_str = f"Target: x={self.target_position[0]:.3f}, y={self.target_position[1]:.3f}, z={self.target_position[2]:.3f}"
            
            # Position error
            if current_pos is not None:
                dx = current_pos[0] - self.target_position[0]
                dy = current_pos[1] - self.target_position[1]
                dz = current_pos[2] - self.target_position[2]
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                error_str = f"Error: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}, dist={distance:.3f}"
            else:
                error_str = "Error: N/A"
            
            # Mission progress
            progress_str = ""
            if self.initial_position is not None and current_pos is not None:
                if self.state == self.STATE_TAKEOFF:
                    initial_alt = self.initial_position[2]
                    target_alt = self.target_position[2]
                    current_alt = current_pos[2]
                    if abs(target_alt - initial_alt) > 0.01:
                        alt_progress = abs((current_alt - initial_alt) / (target_alt - initial_alt)) * 100
                        progress_str = f"Takeoff progress: {alt_progress:.1f}%"
                elif self.state == self.STATE_TRANSLATING:
                    initial_x = self.initial_position[0]
                    target_x = self.target_position[0]
                    current_x = current_pos[0]
                    if abs(target_x - initial_x) > 0.01:
                        trans_progress = abs((current_x - initial_x) / (target_x - initial_x)) * 100
                        progress_str = f"Translation progress: {trans_progress:.1f}%"
                elif self.state == self.STATE_HOVERING:
                    if self.hover_start_time is not None:
                        elapsed_time = (self.get_clock().now() - self.hover_start_time).nanoseconds / 1e9
                        hover_progress = (elapsed_time / self.hover_duration) * 100
                        progress_str = f"Hover progress: {hover_progress:.1f}% ({elapsed_time:.1f}s / {self.hover_duration:.1f}s)"
                elif self.state == self.STATE_LANDING:
                    initial_alt = self.initial_position[2]
                    current_alt = current_pos[2]
                    takeoff_alt = self.initial_position[2] + self.takeoff_height
                    if abs(takeoff_alt - initial_alt) > 0.01:
                        land_progress = abs((current_alt - takeoff_alt) / (initial_alt - takeoff_alt)) * 100
                        progress_str = f"Landing progress: {land_progress:.1f}%"
            
            # Display status
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"State: {state_name} | Nav: {nav_state_name} | Arming: {arming_state_name}")
            self.get_logger().info(f"{pos_str} {pos_valid}")
            self.get_logger().info(f"{target_str}")
            self.get_logger().info(f"{error_str}")
            if progress_str:
                self.get_logger().info(f"{progress_str}")
            self.get_logger().info("=" * 80)

        # State machine logic
        if self.state == self.STATE_INIT:
            # Publish position setpoint to maintain offboard control (hover at current position)
            # Set yaw to 90 degrees for takeoff preparation
            if current_pos is not None:
                self.publish_position_setpoint(
                    current_pos[0],
                    current_pos[1],
                    current_pos[2],
                    yaw=math.pi / 2.0  # 90 degrees in radians
                )
            else:
                # If position not available, publish zero setpoint with 90 degree yaw
                self.publish_position_setpoint(0.0, 0.0, 0.0, yaw=math.pi / 2.0)
            
            # Send offboard mode and arm commands after sending some setpoints
            # Continue sending commands periodically until state changes
            if self.offboard_setpoint_counter >= 10:
                # Send offboard mode command periodically (every 1 second)
                if self.offboard_setpoint_counter % 10 == 0:
                    self.engage_offboard_mode()
                
                # Send arm command periodically (every 1 second, starting after offboard)
                if self.offboard_setpoint_counter >= 20 and self.offboard_setpoint_counter % 10 == 0:
                    self.arm()
            
            if self.offboard_setpoint_counter < 11:
                self.offboard_setpoint_counter += 1
            else:
                # Wait for arming and offboard mode
                # Check arming state: ARMING_STATE_ARMED = 2
                # Also handle case where arming_state might be 0 initially
                is_armed = self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                is_offboard = self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
                
                print(f'arming_state: {self.vehicle_status.arming_state}, nav_state: {self.vehicle_status.nav_state}')
                              
                
                # Log status for debugging
                if self.offboard_setpoint_counter % 50 == 0:  # Every 5 seconds
                    self.get_logger().info(
                        f'Waiting for arming and offboard mode: '
                        f'arming_state={self.vehicle_status.arming_state} (need 2), '
                        f'nav_state={self.vehicle_status.nav_state} (need 14)'
                    )
                
                print(f'is_armed: {is_armed}, is_offboard: {is_offboard}')
                if is_armed and is_offboard:
                    self.get_logger().info('Armed and offboard mode detected. Transitioning to TAKEOFF state.')
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
                
                self.offboard_setpoint_counter += 1

        elif self.state == self.STATE_TAKEOFF:
            # Publish takeoff position setpoint with 90 degree yaw
            self.publish_position_setpoint(
                self.target_position[0],
                self.target_position[1],
                self.target_position[2],
                yaw=math.pi / 2.0  # 90 degrees in radians
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
            # Publish translation position setpoint with 90 degree yaw
            self.publish_position_setpoint(
                self.target_position[0],
                self.target_position[1],
                self.target_position[2],
                yaw=math.pi / 2.0  # 90 degrees in radians
            )
            
            # Check if position reached
            if current_pos is not None:
                if self.is_position_reached(self.target_position, current_pos):
                    self.state = self.STATE_HOVERING
                    self.hover_start_time = self.get_clock().now()
                    self.get_logger().info('Translation complete. Starting hover for 2 seconds.')

        elif self.state == self.STATE_HOVERING:
            # Maintain current position during hover
            self.publish_position_setpoint(
                self.target_position[0],
                self.target_position[1],
                self.target_position[2],
                yaw=math.pi / 2.0  # 90 degrees in radians
            )
            
            # Check if hover duration elapsed
            if self.hover_start_time is not None:
                elapsed_time = (self.get_clock().now() - self.hover_start_time).nanoseconds / 1e9
                if elapsed_time >= self.hover_duration:
                    self.state = self.STATE_LANDING
                    self.get_logger().info('Hover complete. Transitioning to LANDING state.')
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

