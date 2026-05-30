#!/usr/bin/env python3
"""Fill missing TVC joint states so robot_state_publisher can publish all link TFs.

Gazebo JointStatePublisher only lists the gimbal servos by default; the prop
motor joints (motor1_joint, motor2_joint) must also appear in joint_states for
RViz RobotModel to resolve prop_upper/prop_lower transforms.
"""

from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


# Movable joints in models/tvc/tvc.urdf (order matches kinematic tree).
_TVC_JOINTS = (
    'servo0_roll_joint',
    'servo1_pitch_joint',
    'motor1_joint',
    'motor2_joint',
)


class JointStateAdapter(Node):
    def __init__(self) -> None:
        super().__init__('joint_state_adapter')

        self.declare_parameter('input_topic', '/tvc/joint_state')
        self.declare_parameter('output_topic', '/tvc/joint_states')
        self.declare_parameter('default_position', 0.0)

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self._default_position = float(self.get_parameter('default_position').value)

        self._pub = self.create_publisher(JointState, output_topic, 10)
        self.create_subscription(JointState, input_topic, self._on_joint_state, 10)

        self.get_logger().info(
            f'joint_state_adapter: {input_topic} -> {output_topic} '
            f'(joints={list(_TVC_JOINTS)})'
        )

    def _on_joint_state(self, msg: JointState) -> None:
        positions: Dict[str, float] = {}
        for name, pos in zip(msg.name, msg.position):
            positions[name] = float(pos)

        out = JointState()
        out.header = msg.header
        if out.header.stamp.sec == 0 and out.header.stamp.nanosec == 0:
            out.header.stamp = self.get_clock().now().to_msg()

        for joint in _TVC_JOINTS:
            out.name.append(joint)
            out.position.append(positions.get(joint, self._default_position))

        self._pub.publish(out)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = JointStateAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
