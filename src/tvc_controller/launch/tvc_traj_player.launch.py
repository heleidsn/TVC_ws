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
    do_takeoff:=true
    takeoff_altitude_m:=1.0
    off_ground_threshold_m:=0.3
    takeoff_settle_time_s:=1.0
    takeoff_timeout_s:=30.0
    start_velocity_threshold_m_s:=0.1
    time_scale:=1.0
    viz_world_frame:=world
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
        default_value='/home/helei/TVC_ws/TVC-traj-opt/trajs/method_4.csv',
        description='Absolute path to a tvc trajectory CSV exported by tvc_traj_opt_gui.',
    )
    publish_rate_hz = DeclareLaunchArgument(
        'publish_rate_hz', default_value='100.0',
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
    landing_timeout_s = DeclareLaunchArgument(
        'landing_timeout_s', default_value='30.0',
        description='Safety timeout for the LANDING stage [s]. After this, '
                    'force DISARM and shut the node down.',
    )
    use_acc_sp = DeclareLaunchArgument(
        'use_acceleration_setpoint', default_value='true',
        description='Include acceleration in the setpoint (finite-diff of velocity).',
    )
    origin_offset_ned = DeclareLaunchArgument(
        'origin_offset_ned', default_value='[0.0, 0.0, 0.0]',
        description='Optional NED offset [x,y,z] added to the CSV positions (meters).',
    )
    do_takeoff = DeclareLaunchArgument(
        'do_takeoff', default_value='true',
        description='If true, hover at takeoff_altitude_m before replaying the CSV.',
    )
    takeoff_altitude_m = DeclareLaunchArgument(
        'takeoff_altitude_m', default_value='1.0',
        description='Hover altitude (m above ground) used during the TAKEOFF stage.',
    )
    off_ground_threshold_m = DeclareLaunchArgument(
        'off_ground_threshold_m', default_value='0.3',
        description='Altitude (m) above which the vehicle is considered airborne.',
    )
    takeoff_settle_time_s = DeclareLaunchArgument(
        'takeoff_settle_time_s', default_value='1.0',
        description='Minimum hover time after detecting off-ground, before the '
                    'velocity threshold is allowed to trigger playback.',
    )
    takeoff_timeout_s = DeclareLaunchArgument(
        'takeoff_timeout_s', default_value='30.0',
        description='Safety timeout for the TAKEOFF stage [s].',
    )
    start_velocity_threshold_m_s = DeclareLaunchArgument(
        'start_velocity_threshold_m_s', default_value='0.1',
        description='Speed [m/s] below which the vehicle is considered '
                    'settled at the first trajectory point; once reached the '
                    'CSV playback starts.',
    )
    time_scale = DeclareLaunchArgument(
        'time_scale', default_value='2.0',
        description='Trajectory time scaling factor. time_scale=2 doubles the '
                    'playback duration and halves all commanded velocities.',
    )
    viz_world_frame = DeclareLaunchArgument(
        'viz_world_frame', default_value='world',
        description='RViz fixed frame (ENU) used by the planned/executed path publishers.',
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
            'landing_timeout_s': LaunchConfiguration('landing_timeout_s'),
            'use_acceleration_setpoint': LaunchConfiguration('use_acceleration_setpoint'),
            'origin_offset_ned': LaunchConfiguration('origin_offset_ned'),
            'do_takeoff': LaunchConfiguration('do_takeoff'),
            'takeoff_altitude_m': LaunchConfiguration('takeoff_altitude_m'),
            'off_ground_threshold_m': LaunchConfiguration('off_ground_threshold_m'),
            'takeoff_settle_time_s': LaunchConfiguration('takeoff_settle_time_s'),
            'takeoff_timeout_s': LaunchConfiguration('takeoff_timeout_s'),
            'start_velocity_threshold_m_s': LaunchConfiguration('start_velocity_threshold_m_s'),
            'time_scale': LaunchConfiguration('time_scale'),
            'viz_world_frame': LaunchConfiguration('viz_world_frame'),
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
        landing_timeout_s,
        use_acc_sp,
        origin_offset_ned,
        do_takeoff,
        takeoff_altitude_m,
        off_ground_threshold_m,
        takeoff_settle_time_s,
        takeoff_timeout_s,
        start_velocity_threshold_m_s,
        time_scale,
        viz_world_frame,
        use_sim_time,
        log_level,
        player_node,
    ])
