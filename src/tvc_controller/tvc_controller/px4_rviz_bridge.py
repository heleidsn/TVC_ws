#!/usr/bin/env python3
"""
PX4 -> RViz2 visualization bridge.

PX4 publishes its state (via uXRCE-DDS) in NED world frame and FRD body frame,
using its own ``px4_msgs`` types which RViz2 cannot render directly.

This node subscribes to the relevant PX4 topics, performs the standard
PX4 <-> ROS frame transforms (NED <-> ENU for world, FRD <-> FLU for body),
and republishes them as standard ROS 2 messages that RViz2 understands:

    Topic                                   Type                       Source PX4 topic
    --------------------------------------- -------------------------- ------------------------------------
    /px4_visualizer/ekf2_pose                geometry_msgs/PoseStamped  vehicle_local_position + vehicle_attitude (EKF2)
    /px4_visualizer/ekf2_path                nav_msgs/Path              accumulated EKF2 pose
    /px4_visualizer/gz_pose                  geometry_msgs/PoseStamped  Gazebo model odometry
    /px4_visualizer/gz_path                  nav_msgs/Path              accumulated Gazebo pose
    /px4_visualizer/vehicle_pose            geometry_msgs/PoseStamped  alias of ekf2_pose
    /px4_visualizer/vehicle_path            nav_msgs/Path              alias of ekf2_path
    /px4_visualizer/imu                     sensor_msgs/Imu            sensor_combined + vehicle_attitude
    /px4_visualizer/attitude_setpoint_pose  geometry_msgs/PoseStamped  vehicle_attitude_setpoint
    /px4_visualizer/trajectory_setpoint     geometry_msgs/PoseStamped  trajectory_setpoint
    /px4_visualizer/trajectory_setpoint_path nav_msgs/Path             accumulated trajectory setpoints
    /px4_visualizer/home_pose               geometry_msgs/PoseStamped  home_position
    /px4_visualizer/gps_fix                 sensor_msgs/NavSatFix      vehicle_global_position
    /px4_visualizer/velocity_arrow          visualization_msgs/Marker  vehicle_local_position.v[xyz]
    /px4_visualizer/ground_plane            visualization_msgs/Marker  Gazebo world ground (z=0 in world frame)

TF tree published (simulation):
    world -> base_link  from Gazebo /model/tvc_0/odometry (base_link pose via xyz_offset in model.sdf)

PX4 EKF topics are still published for comparison (ekf2_pose/path) but do not drive TF
when Gazebo odometry is available.
"""

import math
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    TransformStamped,
)
from nav_msgs.msg import Odometry, Path
from px4_msgs.msg import (
    HomePosition,
    SensorCombined,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleAttitudeSetpoint,
    VehicleGlobalPosition,
    VehicleLocalPosition,
    VehicleOdometry,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu, NavSatFix
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


# --------------------------------------------------------------------------------------
# Static frame rotations (see PX4 frame_transforms.h)
# --------------------------------------------------------------------------------------
# Rotation that maps a vector expressed in NED into ENU
#   ENU = R_ENU_NED * NED   with R_ENU_NED = [[0,1,0],[1,0,0],[0,0,-1]]
# Equivalent unit quaternion: 180 deg about (1, 1, 0)/sqrt(2).
# scipy uses (x, y, z, w) order.
_NED_TO_ENU = R.from_quat([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])

# Rotation that maps a vector expressed in PX4 body (FRD, "aircraft") into
# ROS body (FLU, "base_link") — a 180 deg rotation about the body x axis.
_AIRCRAFT_TO_BASELINK = R.from_quat([1.0, 0.0, 0.0, 0.0])


def ned_position_to_enu(p_ned: np.ndarray) -> np.ndarray:
    """(north, east, down) -> (east, north, up)."""
    return np.array([p_ned[1], p_ned[0], -p_ned[2]], dtype=float)


def ned_to_enu_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert a PX4 attitude quaternion (NED <- FRD, w,x,y,z order)
    into a ROS attitude quaternion (ENU <- FLU, x,y,z,w order).
    """
    w, x, y, z = q_wxyz
    if abs(w) + abs(x) + abs(y) + abs(z) < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    r_ned_aircraft = R.from_quat([x, y, z, w])
    r_enu_baselink = _NED_TO_ENU * r_ned_aircraft * _AIRCRAFT_TO_BASELINK.inv()
    return r_enu_baselink.as_quat()


def frd_to_flu(v_frd: np.ndarray) -> np.ndarray:
    """Body-frame vector: (forward, right, down) -> (forward, left, up)."""
    return np.array([v_frd[0], -v_frd[1], -v_frd[2]], dtype=float)


# --------------------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------------------
class Px4RvizBridge(Node):
    def __init__(self) -> None:
        super().__init__('px4_rviz_bridge')

        # ---------------- parameters ----------------
        self.declare_parameter('px4_namespace', '')         # e.g. '/px4_1' for multi-vehicle
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('body_frame', 'base_link')
        self.declare_parameter('path_max_length', 5000)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_ground', True)
        self.declare_parameter('ground_size', 500.0)
        self.declare_parameter('gz_odometry_topic', '/model/tvc_0/odometry')

        self._ns: str = self.get_parameter('px4_namespace').value.rstrip('/')
        self._world_frame: str = self.get_parameter('world_frame').value
        self._body_frame: str = self.get_parameter('body_frame').value
        self._path_max_length: int = int(self.get_parameter('path_max_length').value)
        self._publish_tf: bool = bool(self.get_parameter('publish_tf').value)
        self._publish_ground: bool = bool(self.get_parameter('publish_ground').value)
        self._ground_size: float = float(self.get_parameter('ground_size').value)
        self._gz_odometry_topic: str = str(self.get_parameter('gz_odometry_topic').value).strip()
        self._use_gazebo_tf: bool = bool(self._gz_odometry_topic)

        # ---------------- QoS ----------------
        # Match px4_ros_com / lqr_controller_node: PX4 uXRCE-DDS uses
        # BEST_EFFORT + TRANSIENT_LOCAL. VOLATILE here caused DDS history slots
        # to be sized for the first (smaller) sample and later samples fail with
        # "payload size ... larger than the history payload size ... cannot be resized".
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---------------- internal cache ----------------
        # We need attitude to fuse with VehicleLocalPosition (which has no quaternion).
        self._last_attitude_q_wxyz: Optional[np.ndarray] = None
        self._last_attitude_stamp_us: int = 0
        # Track home position for global_position display
        self._home_msg: Optional[HomePosition] = None
        # Path buffers
        self._vehicle_path = Path()
        self._vehicle_path.header.frame_id = self._world_frame
        self._ekf2_path = Path()
        self._ekf2_path.header.frame_id = self._world_frame
        self._gz_path = Path()
        self._gz_path.header.frame_id = self._world_frame
        self._setpoint_path = Path()
        self._setpoint_path.header.frame_id = self._world_frame

        # ---------------- TF broadcaster ----------------
        self._tf_broadcaster = TransformBroadcaster(self)

        # ---------------- subscribers ----------------
        def t(name: str) -> str:
            return f'{self._ns}/fmu/out/{name}' if self._ns else f'/fmu/out/{name}'

        def t_in(name: str) -> str:
            return f'{self._ns}/fmu/in/{name}' if self._ns else f'/fmu/in/{name}'

        self.create_subscription(VehicleAttitude, t('vehicle_attitude'),
                                 self._on_attitude, px4_qos)
        self.create_subscription(VehicleLocalPosition, t('vehicle_local_position'),
                                 self._on_local_position, px4_qos)
        self.create_subscription(VehicleOdometry, t('vehicle_odometry'),
                                 self._on_odometry, px4_qos)
        self.create_subscription(SensorCombined, t('sensor_combined'),
                                 self._on_imu, px4_qos)
        self.create_subscription(VehicleAttitudeSetpoint, t('vehicle_attitude_setpoint'),
                                 self._on_attitude_setpoint, px4_qos)
        self.create_subscription(HomePosition, t('home_position'),
                                 self._on_home_position, px4_qos)
        self.create_subscription(VehicleGlobalPosition, t('vehicle_global_position'),
                                 self._on_global_position, px4_qos)
        # trajectory_setpoint is a PX4 input (offboard command); we display it too
        self.create_subscription(TrajectorySetpoint, t_in('trajectory_setpoint'),
                                 self._on_trajectory_setpoint, px4_qos)

        if self._gz_odometry_topic:
            self.create_subscription(
                Odometry, self._gz_odometry_topic, self._on_gz_odometry, 10)

        # ---------------- publishers ----------------
        self._pub_ekf2_pose = self.create_publisher(
            PoseStamped, '/px4_visualizer/ekf2_pose', 10)
        self._pub_ekf2_path = self.create_publisher(
            Path, '/px4_visualizer/ekf2_path', 10)
        self._pub_gz_pose = self.create_publisher(
            PoseStamped, '/px4_visualizer/gz_pose', 10)
        self._pub_gz_path = self.create_publisher(
            Path, '/px4_visualizer/gz_path', 10)
        self._pub_vehicle_pose = self.create_publisher(
            PoseStamped, '/px4_visualizer/vehicle_pose', 10)
        self._pub_vehicle_odom = self.create_publisher(
            Odometry, '/px4_visualizer/vehicle_odom', 10)
        self._pub_vehicle_path = self.create_publisher(
            Path, '/px4_visualizer/vehicle_path', 10)
        self._pub_imu = self.create_publisher(
            Imu, '/px4_visualizer/imu', 50)
        self._pub_att_sp_pose = self.create_publisher(
            PoseStamped, '/px4_visualizer/attitude_setpoint_pose', 10)
        self._pub_traj_sp_pose = self.create_publisher(
            PoseStamped, '/px4_visualizer/trajectory_setpoint', 10)
        self._pub_traj_sp_path = self.create_publisher(
            Path, '/px4_visualizer/trajectory_setpoint_path', 10)
        self._pub_home_pose = self.create_publisher(
            PoseStamped, '/px4_visualizer/home_pose', 10)
        self._pub_gps_fix = self.create_publisher(
            NavSatFix, '/px4_visualizer/gps_fix', 10)
        self._pub_velocity_arrow = self.create_publisher(
            Marker, '/px4_visualizer/velocity_arrow', 10)
        self._pub_ground_plane = self.create_publisher(
            Marker, '/px4_visualizer/ground_plane', 10)

        if self._publish_ground:
            self._publish_ground_plane()
            self.create_timer(1.0, self._publish_ground_plane)

        self.get_logger().info(
            f'px4_rviz_bridge started (ns="{self._ns or "/"}", world="{self._world_frame}", '
            f'body="{self._body_frame}", tf={self._publish_tf}, '
            f'gazebo_tf={self._use_gazebo_tf}, '
            f'gz_odom="{self._gz_odometry_topic or "disabled"}")'
        )

    # ----------------------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------------------
    def _now_stamp(self):
        return self.get_clock().now().to_msg()

    def _make_pose(self, p_enu: np.ndarray, q_xyzw: np.ndarray) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self._now_stamp()
        msg.header.frame_id = self._world_frame
        msg.pose.position.x = float(p_enu[0])
        msg.pose.position.y = float(p_enu[1])
        msg.pose.position.z = float(p_enu[2])
        msg.pose.orientation.x = float(q_xyzw[0])
        msg.pose.orientation.y = float(q_xyzw[1])
        msg.pose.orientation.z = float(q_xyzw[2])
        msg.pose.orientation.w = float(q_xyzw[3])
        return msg

    def _append_to_path(self, path: Path, pose: PoseStamped, publisher) -> None:
        path.header.stamp = pose.header.stamp
        path.poses.append(pose)
        if len(path.poses) > self._path_max_length:
            del path.poses[: len(path.poses) - self._path_max_length]
        publisher.publish(path)

    def _broadcast_tf(
        self,
        p_enu: np.ndarray,
        q_xyzw: np.ndarray,
        stamp=None,
        world_frame: Optional[str] = None,
        body_frame: Optional[str] = None,
    ) -> None:
        if not self._publish_tf:
            return
        tf = TransformStamped()
        tf.header.stamp = stamp if stamp is not None else self._now_stamp()
        tf.header.frame_id = world_frame or self._world_frame
        tf.child_frame_id = body_frame or self._body_frame
        tf.transform.translation.x = float(p_enu[0])
        tf.transform.translation.y = float(p_enu[1])
        tf.transform.translation.z = float(p_enu[2])
        tf.transform.rotation.x = float(q_xyzw[0])
        tf.transform.rotation.y = float(q_xyzw[1])
        tf.transform.rotation.z = float(q_xyzw[2])
        tf.transform.rotation.w = float(q_xyzw[3])
        self._tf_broadcaster.sendTransform(tf)

    # ----------------------------------------------------------------------------------
    # subscriber callbacks
    # ----------------------------------------------------------------------------------
    def _on_attitude(self, msg: VehicleAttitude) -> None:
        # VehicleAttitude.q is (w, x, y, z) representing NED <- FRD
        self._last_attitude_q_wxyz = np.array(
            [msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=float)
        self._last_attitude_stamp_us = int(msg.timestamp)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        if not msg.xy_valid and not msg.z_valid:
            return
        if self._last_attitude_q_wxyz is None:
            # without attitude we still can show position; use identity orientation
            q_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            q_xyzw = ned_to_enu_quat(self._last_attitude_q_wxyz)

        p_enu = ned_position_to_enu(np.array([msg.x, msg.y, msg.z]))
        pose = self._make_pose(p_enu, q_xyzw)
        self._pub_ekf2_pose.publish(pose)
        self._pub_vehicle_pose.publish(pose)
        self._append_to_path(self._ekf2_path, pose, self._pub_ekf2_path)
        self._append_to_path(self._vehicle_path, pose, self._pub_vehicle_path)
        if not self._use_gazebo_tf:
            self._broadcast_tf(p_enu, q_xyzw)

        # Velocity arrow marker in world frame
        v_enu = ned_position_to_enu(np.array([msg.vx, msg.vy, msg.vz]))
        self._publish_velocity_arrow(p_enu, v_enu)

    def _publish_velocity_arrow(self, p_enu: np.ndarray, v_enu: np.ndarray) -> None:
        arrow = Marker()
        arrow.header.stamp = self._now_stamp()
        arrow.header.frame_id = self._world_frame
        arrow.ns = 'px4_velocity'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        start = Point(x=float(p_enu[0]), y=float(p_enu[1]), z=float(p_enu[2]))
        end = Point(x=float(p_enu[0] + v_enu[0]),
                    y=float(p_enu[1] + v_enu[1]),
                    z=float(p_enu[2] + v_enu[2]))
        arrow.points = [start, end]
        arrow.scale.x = 0.05  # shaft diameter
        arrow.scale.y = 0.12  # head diameter
        arrow.scale.z = 0.15  # head length
        arrow.color.r = 0.1
        arrow.color.g = 0.9
        arrow.color.b = 0.2
        arrow.color.a = 0.9
        self._pub_velocity_arrow.publish(arrow)

    def _publish_ground_plane(self) -> None:
        """Draw ground at z=0 in the Gazebo world frame (matches default.sdf ground_plane)."""
        if not self._publish_ground:
            return
        thickness = 0.01
        ground = Marker()
        ground.header.stamp = self._now_stamp()
        ground.header.frame_id = self._world_frame
        ground.ns = 'ground_plane'
        ground.id = 0
        ground.type = Marker.CUBE
        ground.action = Marker.ADD
        ground.pose.position.x = 0.0
        ground.pose.position.y = 0.0
        ground.pose.position.z = -thickness * 0.5
        ground.pose.orientation.w = 1.0
        ground.scale.x = self._ground_size
        ground.scale.y = self._ground_size
        ground.scale.z = thickness
        ground.color.r = 0.78
        ground.color.g = 0.78
        ground.color.b = 0.78
        ground.color.a = 1.0
        self._pub_ground_plane.publish(ground)

    def _on_gz_odometry(self, msg: Odometry) -> None:
        # Always relabel the Gazebo odometry to our configured world frame.
        # Gazebo Sim publishes /model/<name>/odometry with frame_id="odom" by
        # default, but the rest of the visualisation pipeline (rviz Fixed
        # Frame, paths, ground-plane marker, TF tree) is anchored at the
        # configured world_frame (default "world"). Letting "odom" propagate
        # would split the TF tree and cause rviz Message Filter queues to
        # overflow ("dropping message: frame 'odom' ... queue is full").
        world_frame = self._world_frame
        incoming_body = msg.child_frame_id.strip()
        body_frame = incoming_body if incoming_body else self._body_frame

        p_enu = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=float)
        if not np.all(np.isfinite(p_enu)):
            return
        q_xyzw = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=float)
        if not np.all(np.isfinite(q_xyzw)) or np.linalg.norm(q_xyzw) < 1e-6:
            q_xyzw = np.array([0.0, 0.0, 0.0, 1.0])

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = world_frame
        pose.pose = msg.pose.pose
        self._pub_gz_pose.publish(pose)
        self._append_to_path(self._gz_path, pose, self._pub_gz_path)
        self._broadcast_tf(
            p_enu,
            q_xyzw,
            stamp=msg.header.stamp,
            world_frame=world_frame,
            body_frame=body_frame,
        )

    def _on_odometry(self, msg: VehicleOdometry) -> None:
        # Pose
        p_ned = np.array([msg.position[0], msg.position[1], msg.position[2]])
        if not np.all(np.isfinite(p_ned)):
            return
        q_wxyz = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
        if not np.all(np.isfinite(q_wxyz)) or np.linalg.norm(q_wxyz) < 1e-6:
            return

        p_enu = ned_position_to_enu(p_ned)
        q_xyzw = ned_to_enu_quat(q_wxyz)

        odom = Odometry()
        odom.header.stamp = self._now_stamp()
        odom.header.frame_id = self._world_frame
        odom.child_frame_id = self._body_frame
        odom.pose.pose.position.x = float(p_enu[0])
        odom.pose.pose.position.y = float(p_enu[1])
        odom.pose.pose.position.z = float(p_enu[2])
        odom.pose.pose.orientation.x = float(q_xyzw[0])
        odom.pose.pose.orientation.y = float(q_xyzw[1])
        odom.pose.pose.orientation.z = float(q_xyzw[2])
        odom.pose.pose.orientation.w = float(q_xyzw[3])

        # Linear velocity (nav_msgs/Odometry.twist is expressed in child_frame_id = base_link = FLU)
        v_in = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])
        if np.all(np.isfinite(v_in)):
            if msg.velocity_frame in (
                VehicleOdometry.VELOCITY_FRAME_NED,
                VehicleOdometry.VELOCITY_FRAME_FRD,   # world-fixed FRD, treat as NED for viz
            ):
                v_world_enu = ned_position_to_enu(v_in)
                r_world_body = R.from_quat(q_xyzw)
                v_body_flu = r_world_body.inv().apply(v_world_enu)
            elif msg.velocity_frame == VehicleOdometry.VELOCITY_FRAME_BODY_FRD:
                v_body_flu = frd_to_flu(v_in)
            else:  # UNKNOWN — fall back to body FRD
                v_body_flu = frd_to_flu(v_in)
            odom.twist.twist.linear.x = float(v_body_flu[0])
            odom.twist.twist.linear.y = float(v_body_flu[1])
            odom.twist.twist.linear.z = float(v_body_flu[2])

        # Angular velocity (FRD body) -> FLU body
        w_in = np.array([msg.angular_velocity[0],
                         msg.angular_velocity[1],
                         msg.angular_velocity[2]])
        if np.all(np.isfinite(w_in)):
            w_flu = frd_to_flu(w_in)
            odom.twist.twist.angular.x = float(w_flu[0])
            odom.twist.twist.angular.y = float(w_flu[1])
            odom.twist.twist.angular.z = float(w_flu[2])

        self._pub_vehicle_odom.publish(odom)

    def _on_imu(self, msg: SensorCombined) -> None:
        if self._last_attitude_q_wxyz is None:
            return
        q_xyzw = ned_to_enu_quat(self._last_attitude_q_wxyz)
        imu = Imu()
        imu.header.stamp = self._now_stamp()
        imu.header.frame_id = self._body_frame
        imu.orientation.x = float(q_xyzw[0])
        imu.orientation.y = float(q_xyzw[1])
        imu.orientation.z = float(q_xyzw[2])
        imu.orientation.w = float(q_xyzw[3])

        # FRD -> FLU
        w_flu = frd_to_flu(np.array(msg.gyro_rad))
        imu.angular_velocity.x = float(w_flu[0])
        imu.angular_velocity.y = float(w_flu[1])
        imu.angular_velocity.z = float(w_flu[2])

        a_flu = frd_to_flu(np.array(msg.accelerometer_m_s2))
        imu.linear_acceleration.x = float(a_flu[0])
        imu.linear_acceleration.y = float(a_flu[1])
        imu.linear_acceleration.z = float(a_flu[2])
        self._pub_imu.publish(imu)

    def _on_attitude_setpoint(self, msg: VehicleAttitudeSetpoint) -> None:
        q_wxyz = np.array([msg.q_d[0], msg.q_d[1], msg.q_d[2], msg.q_d[3]])
        if np.linalg.norm(q_wxyz) < 1e-6:
            return
        q_xyzw = ned_to_enu_quat(q_wxyz)
        # Anchor the orientation arrow at the latest known vehicle position if any,
        # else at world origin.
        if self._ekf2_path.poses:
            last = self._ekf2_path.poses[-1].pose.position
            p_enu = np.array([last.x, last.y, last.z])
        elif self._vehicle_path.poses:
            last = self._vehicle_path.poses[-1].pose.position
            p_enu = np.array([last.x, last.y, last.z])
        else:
            p_enu = np.zeros(3)
        pose = self._make_pose(p_enu, q_xyzw)
        self._pub_att_sp_pose.publish(pose)

    def _on_trajectory_setpoint(self, msg: TrajectorySetpoint) -> None:
        p_ned = np.array(msg.position, dtype=float)
        if not np.all(np.isfinite(p_ned)):
            return
        p_enu = ned_position_to_enu(p_ned)
        # yaw (NED, radians from north, clockwise positive) -> ENU yaw (from east, CCW positive)
        yaw_ned = msg.yaw if math.isfinite(msg.yaw) else 0.0
        yaw_enu = math.pi / 2.0 - yaw_ned
        q_xyzw = R.from_euler('z', yaw_enu).as_quat()
        pose = self._make_pose(p_enu, q_xyzw)
        self._pub_traj_sp_pose.publish(pose)
        self._append_to_path(self._setpoint_path, pose, self._pub_traj_sp_path)

    def _on_home_position(self, msg: HomePosition) -> None:
        self._home_msg = msg
        # home_position publishes the home as NED relative to local origin (x, y, z)
        p_enu = ned_position_to_enu(np.array([msg.x, msg.y, msg.z]))
        yaw_enu = math.pi / 2.0 - (msg.yaw if math.isfinite(msg.yaw) else 0.0)
        q_xyzw = R.from_euler('z', yaw_enu).as_quat()
        self._pub_home_pose.publish(self._make_pose(p_enu, q_xyzw))

    def _on_global_position(self, msg: VehicleGlobalPosition) -> None:
        fix = NavSatFix()
        fix.header.stamp = self._now_stamp()
        fix.header.frame_id = self._body_frame
        fix.latitude = float(msg.lat)
        fix.longitude = float(msg.lon)
        fix.altitude = float(msg.alt)
        self._pub_gps_fix.publish(fix)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = Px4RvizBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
