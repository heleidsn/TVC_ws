#!/usr/bin/env python3
"""
Publish Gazebo model odometry to PX4 as external vision (vehicle_visual_odometry).

Subscribes to Gazebo ground truth in ROS ENU / body FLU (nav_msgs/Odometry),
converts to PX4 NED / body FRD (px4_msgs/VehicleOdometry), and publishes to
/fmu/in/vehicle_visual_odometry for EKF2 fusion.

Frame conversion matches PX4 GZBridge::odometryCallback().

Note: OdometryPublisher reports pose in the model ``odom`` frame (relative to
spawn), NOT absolute Gazebo world coordinates. At rest on the pad the position
is expected to be (0, 0, 0); it changes when the vehicle moves.
"""

import math
from typing import List, Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from scipy.spatial.transform import Rotation as R


_ENU_TO_NED = R.from_quat([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]).inv()
_FLU_TO_FRD = R.from_quat([1.0, 0.0, 0.0, 0.0])


def enu_position_to_ned(p_enu: np.ndarray) -> np.ndarray:
    """(east, north, up) -> (north, east, down)."""
    return np.array([p_enu[1], p_enu[0], -p_enu[2]], dtype=float)


def flu_to_frd(v_flu: np.ndarray) -> np.ndarray:
    """Body vector: FLU -> FRD."""
    return np.array([v_flu[0], -v_flu[1], -v_flu[2]], dtype=float)


def flu_enu_to_frd_ned_quat(q_flu_enu_xyzw: np.ndarray) -> np.ndarray:
    """Convert body FLU->ENU quaternion to PX4 body FRD->NED (w, x, y, z)."""
    r_flu_enu = R.from_quat(q_flu_enu_xyzw)
    r_frd_ned = _ENU_TO_NED * r_flu_enu * _FLU_TO_FRD.inv()
    q_xyzw = r_frd_ned.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


class GzVisionPublisher(Node):
    def __init__(self) -> None:
        super().__init__('gz_vision_publisher')

        self.declare_parameter('px4_namespace', '')
        self.declare_parameter('gz_odometry_topic', '/model/tvc_0/odometry_with_covariance')
        self.declare_parameter('position_variance', 0.01)
        self.declare_parameter('orientation_variance', 0.01)
        self.declare_parameter('velocity_variance', 0.01)
        self.declare_parameter('quality', 100)

        ns = str(self.get_parameter('px4_namespace').value).rstrip('/')
        gz_topic = str(self.get_parameter('gz_odometry_topic').value).strip()
        self._position_var = float(self.get_parameter('position_variance').value)
        self._orientation_var = float(self.get_parameter('orientation_variance').value)
        self._velocity_var = float(self.get_parameter('velocity_variance').value)
        self._quality = int(self.get_parameter('quality').value)

        px4_in_topic = (
            f'{ns}/fmu/in/vehicle_visual_odometry'
            if ns else '/fmu/in/vehicle_visual_odometry'
        )

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._pub_visual_odom = self.create_publisher(
            VehicleOdometry, px4_in_topic, px4_qos)

        if not gz_topic:
            self.get_logger().error('gz_odometry_topic is empty; nothing to subscribe to')
            return

        self.create_subscription(Odometry, gz_topic, self._on_gz_odometry, 10)

        self._debug_counter = 0
        self.get_logger().info(
            f'gz_vision_publisher: {gz_topic} -> {px4_in_topic} '
            f'(odom-relative ENU -> PX4 local NED)'
        )

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds // 1000)

    def _on_gz_odometry(self, msg: Odometry) -> None:
        p_enu = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=float)
        if not np.all(np.isfinite(p_enu)):
            return

        q_flu_enu = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=float)
        if not np.all(np.isfinite(q_flu_enu)) or np.linalg.norm(q_flu_enu) < 1e-6:
            return

        p_ned = enu_position_to_ned(p_enu)
        q_wxyz = flu_enu_to_frd_ned_quat(q_flu_enu)

        out = VehicleOdometry()
        stamp_us = self._now_us()
        out.timestamp = stamp_us
        out.timestamp_sample = stamp_us

        out.pose_frame = VehicleOdometry.POSE_FRAME_NED
        out.position = [float(p_ned[0]), float(p_ned[1]), float(p_ned[2])]
        out.q = [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])]

        out.velocity_frame = VehicleOdometry.VELOCITY_FRAME_BODY_FRD
        v_flu = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ], dtype=float)
        w_flu = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ], dtype=float)
        if np.all(np.isfinite(v_flu)):
            v_frd = flu_to_frd(v_flu)
            out.velocity = [float(v_frd[0]), float(v_frd[1]), float(v_frd[2])]
        if np.all(np.isfinite(w_flu)):
            w_frd = flu_to_frd(w_flu)
            out.angular_velocity = [float(w_frd[0]), float(w_frd[1]), float(w_frd[2])]

        out.position_variance = [
            self._position_var, self._position_var, self._position_var]
        out.orientation_variance = [
            self._orientation_var, self._orientation_var, self._orientation_var]
        out.velocity_variance = [
            self._velocity_var, self._velocity_var, self._velocity_var]
        out.quality = self._quality

        self._pub_visual_odom.publish(out)

        self._debug_counter += 1
        if self._debug_counter % 50 == 0:
            self.get_logger().info(
                f'gz odom ENU [{p_enu[0]:.3f}, {p_enu[1]:.3f}, {p_enu[2]:.3f}] -> '
                f'vision NED [{p_ned[0]:.3f}, {p_ned[1]:.3f}, {p_ned[2]:.3f}]'
            )


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = GzVisionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
