"""Launch the PX4 -> RViz2 bridge together with RViz2."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_robot_description(package_dir: str) -> str:
    candidates = [
        os.path.join(package_dir, 'models', 'tvc', 'tvc.urdf'),
        os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'models', 'tvc', 'tvc.urdf')),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as urdf_file:
                return urdf_file.read()
    raise FileNotFoundError('tvc.urdf not found in package share or source tree')


def generate_launch_description() -> LaunchDescription:
    package_name = 'tvc_controller'
    package_dir = get_package_share_directory(package_name)
    default_rviz_config = os.path.join(package_dir, 'config', 'px4_rviz.rviz')
    robot_description = _load_robot_description(package_dir)

    px4_namespace_arg = DeclareLaunchArgument(
        'px4_namespace',
        default_value='',
        description='PX4 topic namespace prefix (e.g. "/px4_1"). Empty for default single vehicle.',
    )
    world_frame_arg = DeclareLaunchArgument(
        'world_frame', default_value='world',
        description='ROS world frame id (ENU, matches Gazebo world).',
    )
    body_frame_arg = DeclareLaunchArgument(
        'body_frame', default_value='base_link',
        description='ROS body frame id (FLU).',
    )
    publish_tf_arg = DeclareLaunchArgument(
        'publish_tf', default_value='true',
        description='Whether the bridge should broadcast world -> base_link TF.',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation clock (set true when running with PX4 SITL).',
    )
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz', default_value='true',
        description='Whether to start RViz2 with the pre-canned config.',
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz_config,
        description='Path to the RViz2 configuration file.',
    )
    gz_odometry_topic_arg = DeclareLaunchArgument(
        'gz_odometry_topic',
        default_value='/model/tvc_0/odometry',
        description='Gazebo ground-truth odometry topic (ENU). Empty string disables.',
    )
    launch_gz_vision_arg = DeclareLaunchArgument(
        'launch_gz_vision',
        default_value='true',
        description='Publish Gazebo odometry to PX4 vehicle_visual_odometry.',
    )
    launch_robot_model_arg = DeclareLaunchArgument(
        'launch_robot_model', default_value='true',
        description='Start robot_state_publisher for RViz RobotModel display.',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')

    bridge_node = Node(
        package=package_name,
        executable='px4_rviz_bridge',
        name='px4_rviz_bridge',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'px4_namespace': LaunchConfiguration('px4_namespace'),
            'world_frame': LaunchConfiguration('world_frame'),
            'body_frame': LaunchConfiguration('body_frame'),
            'publish_tf': LaunchConfiguration('publish_tf'),
            'gz_odometry_topic': LaunchConfiguration('gz_odometry_topic'),
            'use_sim_time': use_sim_time,
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
    )

    gz_vision_node = Node(
        package=package_name,
        executable='gz_vision_publisher',
        name='gz_vision_publisher',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'px4_namespace': LaunchConfiguration('px4_namespace'),
            'gz_odometry_topic': LaunchConfiguration('gz_odometry_topic'),
            'use_sim_time': use_sim_time,
        }],
        condition=IfCondition(LaunchConfiguration('launch_gz_vision')),
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
        remappings=[('/joint_states', '/tvc/joint_state')],
        condition=IfCondition(LaunchConfiguration('launch_robot_model')),
    )

    return LaunchDescription([
        px4_namespace_arg,
        world_frame_arg,
        body_frame_arg,
        publish_tf_arg,
        use_sim_time_arg,
        launch_rviz_arg,
        rviz_config_arg,
        gz_odometry_topic_arg,
        launch_gz_vision_arg,
        launch_robot_model_arg,
        bridge_node,
        gz_vision_node,
        robot_state_publisher_node,
        rviz_node,
    ])
