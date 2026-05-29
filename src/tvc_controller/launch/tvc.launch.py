"""Unified launch file for TVC simulation stack."""

import os
import shutil

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_px4_dir(package_dir: str) -> str:
    """Resolve PX4-TVC-NUS path from install layout or source tree."""
    candidates = [
        os.path.abspath(os.path.join(package_dir, '..', '..', '..', '..', 'PX4-TVC-NUS')),
        os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'PX4-TVC-NUS')),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _find_microxrce_agent() -> str:
    for name in ('MicroXRCEAgent', 'micro-xrce-dds-agent'):
        path = shutil.which(name)
        if path:
            return path
    return 'MicroXRCEAgent'


def _load_robot_description(package_dir: str) -> str:
    """Load TVC URDF from install share or source tree."""
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
    default_config_file = os.path.join(package_dir, 'config', 'tvc_params.yaml')
    default_rviz_config = os.path.join(package_dir, 'config', 'px4_rviz.rviz')
    default_bridge_config = os.path.join(package_dir, 'config', 'bridge.yaml')
    default_px4_dir = _default_px4_dir(package_dir)
    robot_description = _load_robot_description(package_dir)
    default_px4_binary = os.path.join(
        default_px4_dir, 'build', 'px4_sitl_default', 'bin', 'px4')
    default_microxrce_agent = _find_microxrce_agent()

    # Fast DDS: allow history buffers to grow when PX4 message payloads vary
    # (avoids RTPS_READER_HISTORY "cannot be resized" errors on /fmu/out/*).
    # Search candidates in this order: installed share dir, src tree (so the
    # profile still works when the user forgets to re-run `colcon build`).
    fastrtps_profile_candidates = [
        os.path.join(package_dir, 'fastrtps_profile.xml'),
        os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'fastrtps_profile.xml')),
        os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..',
            'src', 'tvc_controller', 'fastrtps_profile.xml')),
    ]
    fastrtps_profile = next(
        (p for p in fastrtps_profile_candidates if os.path.isfile(p)),
        fastrtps_profile_candidates[0],
    )
    set_fastrtps_profile = SetEnvironmentVariable(
        'FASTRTPS_DEFAULT_PROFILES_FILE',
        fastrtps_profile,
    )
    # The XML profile is Fast DDS specific; force the rmw to fastrtps so the
    # historyMemoryPolicy=DYNAMIC setting is actually honored.
    set_rmw_impl = SetEnvironmentVariable(
        'RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp',
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Path to the LQR YAML configuration file.',
    )
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
    log_level_arg = DeclareLaunchArgument(
        'log_level', default_value='info',
        description='Log level for the LQR controller (debug, info, warn, error).',
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz_config,
        description='Path to the RViz2 configuration file.',
    )
    gz_bridge_config_arg = DeclareLaunchArgument(
        'gz_bridge_config', default_value=default_bridge_config,
        description='Path to the ros_gz_bridge YAML configuration file.',
    )
    gz_odometry_topic_arg = DeclareLaunchArgument(
        'gz_odometry_topic',
        default_value='/model/tvc_0/odometry',
        description='Gazebo ground-truth odometry topic (ENU). Empty string disables.',
    )
    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir', default_value=default_px4_dir,
        description='Path to the PX4-TVC-NUS repository.',
    )
    px4_binary_arg = DeclareLaunchArgument(
        'px4_binary', default_value=default_px4_binary,
        description='Path to the PX4 SITL binary.',
    )
    px4_sys_autostart_arg = DeclareLaunchArgument(
        'px4_sys_autostart', default_value='6003',
        description='PX4 SYS_AUTOSTART airframe id.',
    )
    px4_sim_model_arg = DeclareLaunchArgument(
        'px4_sim_model', default_value='tvc',
        description='PX4 Gazebo model name.',
    )
    px4_gz_world_arg = DeclareLaunchArgument(
        'px4_gz_world', default_value='default',
        description='PX4 Gazebo world name.',
    )
    microxrce_agent_arg = DeclareLaunchArgument(
        'microxrce_agent', default_value=default_microxrce_agent,
        description='MicroXRCE-DDS agent executable path.',
    )
    microxrce_port_arg = DeclareLaunchArgument(
        'microxrce_port', default_value='8888',
        description='UDP port for the MicroXRCE-DDS agent.',
    )
    ros_startup_delay_arg = DeclareLaunchArgument(
        'ros_startup_delay', default_value='5.0',
        description='Seconds to wait before starting ROS nodes (allows PX4/Gazebo to initialize).',
    )
    launch_px4_sitl_arg = DeclareLaunchArgument(
        'launch_px4_sitl', default_value='true',
        description='Start PX4 SITL with Gazebo.',
    )
    launch_microxrce_agent_arg = DeclareLaunchArgument(
        'launch_microxrce_agent', default_value='true',
        description='Start the MicroXRCE-DDS agent.',
    )
    launch_controller_arg = DeclareLaunchArgument(
        'launch_controller', default_value='true',
        description='Start the LQR PX4 controller node.',
    )
    launch_gz_bridge_arg = DeclareLaunchArgument(
        'launch_gz_bridge', default_value='true',
        description='Start ros_gz_bridge (clock + Gazebo odometry).',
    )
    launch_px4_rviz_bridge_arg = DeclareLaunchArgument(
        'launch_px4_rviz_bridge', default_value='true',
        description='Start the PX4 -> RViz2 visualization bridge.',
    )
    launch_gz_vision_arg = DeclareLaunchArgument(
        'launch_gz_vision', default_value='true',
        description='Publish Gazebo odometry to PX4 vehicle_visual_odometry.',
    )
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz', default_value='true',
        description='Start RViz2 with the pre-canned config.',
    )
    launch_robot_model_arg = DeclareLaunchArgument(
        'launch_robot_model', default_value='true',
        description='Start robot_state_publisher for RViz RobotModel display.',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')

    microxrce_agent_process = ExecuteProcess(
        cmd=[
            LaunchConfiguration('microxrce_agent'),
            'udp4',
            '-p', LaunchConfiguration('microxrce_port'),
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_microxrce_agent')),
    )

    px4_sitl_process = ExecuteProcess(
        cmd=[LaunchConfiguration('px4_binary')],
        cwd=LaunchConfiguration('px4_dir'),
        additional_env={
            'PX4_SYS_AUTOSTART': LaunchConfiguration('px4_sys_autostart'),
            'PX4_SIM_MODEL': LaunchConfiguration('px4_sim_model'),
            'PX4_GZ_WORLD': LaunchConfiguration('px4_gz_world'),
        },
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_px4_sitl')),
    )

    gz_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'config_file': LaunchConfiguration('gz_bridge_config'),
            'use_sim_time': use_sim_time,
        }],
        condition=IfCondition(LaunchConfiguration('launch_gz_bridge')),
    )

    lqr_controller_node = Node(
        package=package_name,
        executable='lqr_px4_controller',
        name='lqr_px4_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': use_sim_time},
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('launch_controller')),
    )

    px4_rviz_bridge_node = Node(
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
        condition=IfCondition(LaunchConfiguration('launch_px4_rviz_bridge')),
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

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
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

    ros_stack = TimerAction(
        period=LaunchConfiguration('ros_startup_delay'),
        actions=[
            gz_bridge_node,
            # lqr_controller_node,
            robot_state_publisher_node,
            px4_rviz_bridge_node,
            gz_vision_node,
            rviz_node,
        ],
    )

    return LaunchDescription([
        set_fastrtps_profile,
        set_rmw_impl,
        config_file_arg,
        px4_namespace_arg,
        world_frame_arg,
        body_frame_arg,
        publish_tf_arg,
        use_sim_time_arg,
        log_level_arg,
        rviz_config_arg,
        gz_bridge_config_arg,
        gz_odometry_topic_arg,
        px4_dir_arg,
        px4_binary_arg,
        px4_sys_autostart_arg,
        px4_sim_model_arg,
        px4_gz_world_arg,
        microxrce_agent_arg,
        microxrce_port_arg,
        ros_startup_delay_arg,
        launch_px4_sitl_arg,
        launch_microxrce_agent_arg,
        launch_controller_arg,
        launch_gz_bridge_arg,
        launch_px4_rviz_bridge_arg,
        launch_gz_vision_arg,
        launch_rviz_arg,
        launch_robot_model_arg,
        microxrce_agent_process,
        px4_sitl_process,
        ros_stack,
    ])
