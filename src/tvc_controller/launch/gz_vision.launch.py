"""Publish Gazebo odometry to PX4 as vehicle_visual_odometry."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    px4_namespace_arg = DeclareLaunchArgument(
        'px4_namespace',
        default_value='',
        description='PX4 topic namespace prefix (e.g. "/px4_1").',
    )
    gz_odometry_topic_arg = DeclareLaunchArgument(
        'gz_odometry_topic',
        default_value='/model/tvc_0/odometry_with_covariance',
        description='Gazebo ground-truth odometry topic (ENU).',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock.',
    )

    return LaunchDescription([
        px4_namespace_arg,
        gz_odometry_topic_arg,
        use_sim_time_arg,
        Node(
            package='tvc_controller',
            executable='gz_vision_publisher',
            name='gz_vision_publisher',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'px4_namespace': LaunchConfiguration('px4_namespace'),
                'gz_odometry_topic': LaunchConfiguration('gz_odometry_topic'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
        ),
    ])
