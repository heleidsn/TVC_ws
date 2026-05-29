"""Launch file for the TVC trajectory player.

Usage::

    ros2 launch tvc_controller tvc_traj_player.launch.py \
        csv_path:=/path/to/tvc_traj_xxx.csv

Optional arguments::

    publish_rate_hz:=50.0
    arm_wait_setpoints:=20
    hold_after_done_s:=2.0
    land_after_done:=true
    use_acceleration_setpoint:=true
    origin_offset_ned:="[0.0, 0.0, 0.0]"
    use_sim_time:=true
    log_level:=info
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    csv_path = DeclareLaunchArgument(
        'csv_path',
        description='Absolute path to a tvc trajectory CSV exported by tvc_traj_opt_gui.',
    )
    publish_rate_hz = DeclareLaunchArgument(
        'publish_rate_hz', default_value='50.0',
        description='Trajectory setpoint publish rate [Hz]. PX4 needs >= 2 Hz.',
    )
    arm_wait_setpoints = DeclareLaunchArgument(
        'arm_wait_setpoints', default_value='20',
        description='Number of initial setpoints to stream before requesting ARM/OFFBOARD.',
    )
    hold_after_done_s = DeclareLaunchArgument(
        'hold_after_done_s', default_value='2.0',
        description='Seconds to hold the final pose after the CSV ends, before LAND.',
    )
    land_after_done = DeclareLaunchArgument(
        'land_after_done', default_value='true',
        description='Whether to issue a LAND command after the hold phase.',
    )
    use_acc_sp = DeclareLaunchArgument(
        'use_acceleration_setpoint', default_value='true',
        description='Include acceleration in the setpoint (finite-diff of velocity).',
    )
    origin_offset_ned = DeclareLaunchArgument(
        'origin_offset_ned', default_value='[0.0, 0.0, 0.0]',
        description='Optional NED offset [x,y,z] added to the CSV positions (meters).',
    )
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time if true.',
    )
    log_level = DeclareLaunchArgument(
        'log_level', default_value='info',
        description='Log level (debug, info, warn, error).',
    )

    player_node = Node(
        package='tvc_controller',
        executable='tvc_traj_player',
        name='tvc_traj_player',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'csv_path': LaunchConfiguration('csv_path'),
            'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
            'arm_wait_setpoints': LaunchConfiguration('arm_wait_setpoints'),
            'hold_after_done_s': LaunchConfiguration('hold_after_done_s'),
            'land_after_done': LaunchConfiguration('land_after_done'),
            'use_acceleration_setpoint': LaunchConfiguration('use_acceleration_setpoint'),
            'origin_offset_ned': LaunchConfiguration('origin_offset_ned'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )

    return LaunchDescription([
        csv_path,
        publish_rate_hz,
        arm_wait_setpoints,
        hold_after_done_s,
        land_after_done,
        use_acc_sp,
        origin_offset_ned,
        use_sim_time,
        log_level,
        player_node,
    ])
