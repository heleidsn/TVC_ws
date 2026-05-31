"""Launch file for the TVC mission player (trajectory CSV or waypoint sequence).

Play modes::

    play_mode:=trajectory   # CSV offboard playback (default)
    play_mode:=waypoint     # GotoSetpoint in OFFBOARD (no GPS required)

Usage::

    ros2 launch tvc_controller tvc_traj_player.launch.py play_mode:=waypoint

    ros2 launch tvc_controller tvc_traj_player.launch.py \\
        play_mode:=trajectory csv_path:=/path/to/traj.csv
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    play_mode = DeclareLaunchArgument(
        'play_mode', default_value='waypoint',
        description='trajectory: CSV offboard; waypoint: GotoSetpoint in OFFBOARD.',
    )
    csv_path = DeclareLaunchArgument(
        'csv_path',
        default_value='/home/helei/TVC_ws/TVC-traj-opt/trajs/method_4.csv',
        description='Absolute path to trajectory CSV (trajectory mode only).',
    )
    waypoints_enu = DeclareLaunchArgument(
        'waypoints_enu',
        default_value=(
            '0,0,1.5,0,2;'
            '1,0,1.5,0,2;'
            '1,1,1.5,0,2;'
            '-1,1,2.5,0,2;'
            '-1,0,1.5,0,2;'
            '0,0,1.5,0,2'
        ),
        description='Waypoint mode: ENU x,y,z,yaw_deg,hover_s rows separated by ";".',
    )
    publish_rate_hz = DeclareLaunchArgument(
        'publish_rate_hz', default_value='100.0',
        description='Control / goto publish rate [Hz]. GotoSetpoint needs >= 2 Hz.',
    )
    arm_wait_setpoints = DeclareLaunchArgument(
        'arm_wait_setpoints', default_value='20',
        description='Ticks/setpoints before requesting ARM.',
    )
    hold_after_traj_s = DeclareLaunchArgument(
        'hold_after_traj_s', default_value='2.0',
        description='Trajectory mode: hold final CSV pose before return home [s].',
    )
    hover_before_play_s = DeclareLaunchArgument(
        'hover_before_play_s', default_value='2.0',
        description='Trajectory mode: hover at CSV[0] before playback [s].',
    )
    home_hover_altitude_m = DeclareLaunchArgument(
        'home_hover_altitude_m', default_value='1.0',
        description='Trajectory mode: home hover altitude [m] ENU.',
    )
    land_after_done = DeclareLaunchArgument(
        'land_after_done', default_value='true',
        description='Whether to LAND after mission completes.',
    )
    landing_timeout_s = DeclareLaunchArgument(
        'landing_timeout_s', default_value='30.0',
        description='Safety timeout for LANDING [s].',
    )
    waypoint_timeout_s = DeclareLaunchArgument(
        'waypoint_timeout_s', default_value='120.0',
        description='Waypoint mode: per-waypoint timeout [s].',
    )
    use_acc_sp = DeclareLaunchArgument(
        'use_acceleration_setpoint', default_value='true',
        description='Trajectory mode: include acceleration in offboard setpoints.',
    )
    origin_offset_ned = DeclareLaunchArgument(
        'origin_offset_ned', default_value='[0.0, 0.0, 0.0]',
        description='Optional NED offset added to setpoints [x,y,z] (m).',
    )
    off_ground_threshold_m = DeclareLaunchArgument(
        'off_ground_threshold_m', default_value='0.3',
        description='Trajectory mode: airborne altitude threshold [m] (logging).',
    )
    takeoff_timeout_s = DeclareLaunchArgument(
        'takeoff_timeout_s', default_value='30.0',
        description='Trajectory mode: TAKEOFF timeout [s].',
    )
    approach_first_timeout_s = DeclareLaunchArgument(
        'approach_first_timeout_s', default_value='60.0',
        description='Trajectory mode: fly-to-CSV[0] timeout [s].',
    )
    return_timeout_s = DeclareLaunchArgument(
        'return_timeout_s', default_value='60.0',
        description='Trajectory mode: return-home timeout [s].',
    )
    start_velocity_threshold_m_s = DeclareLaunchArgument(
        'start_velocity_threshold_m_s', default_value='0.1',
        description='Speed [m/s] below which a target is considered reached.',
    )
    start_position_threshold_m = DeclareLaunchArgument(
        'start_position_threshold_m', default_value='0.5',
        description='Max distance [m] to home/return targets before advancing.',
    )
    first_point_position_threshold_m = DeclareLaunchArgument(
        'first_point_position_threshold_m', default_value='0.1',
        description='Trajectory mode: |Δp| at CSV[0] must be below this before hover/play.',
    )
    first_point_velocity_threshold_m_s = DeclareLaunchArgument(
        'first_point_velocity_threshold_m_s', default_value='0.1',
        description='Trajectory mode: |v| at CSV[0] must be below this before hover/play.',
    )
    time_scale = DeclareLaunchArgument(
        'time_scale', default_value='2.0',
        description='Trajectory mode: CSV time scaling factor.',
    )
    viz_world_frame = DeclareLaunchArgument(
        'viz_world_frame', default_value='world',
        description='RViz fixed frame (ENU) for planned path visualization.',
    )
    viz_waypoints_enu = DeclareLaunchArgument(
        'viz_waypoints_enu',
        default_value='',
        description=(
            'Optional ENU waypoints for RViz markers only: '
            'x,y,z,yaw_deg rows separated by ";".'
        ),
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
            'play_mode': LaunchConfiguration('play_mode'),
            'csv_path': LaunchConfiguration('csv_path'),
            'waypoints_enu': LaunchConfiguration('waypoints_enu'),
            'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
            'arm_wait_setpoints': LaunchConfiguration('arm_wait_setpoints'),
            'hold_after_traj_s': LaunchConfiguration('hold_after_traj_s'),
            'hover_before_play_s': LaunchConfiguration('hover_before_play_s'),
            'home_hover_altitude_m': LaunchConfiguration('home_hover_altitude_m'),
            'land_after_done': LaunchConfiguration('land_after_done'),
            'landing_timeout_s': LaunchConfiguration('landing_timeout_s'),
            'waypoint_timeout_s': LaunchConfiguration('waypoint_timeout_s'),
            'use_acceleration_setpoint': LaunchConfiguration('use_acceleration_setpoint'),
            'origin_offset_ned': LaunchConfiguration('origin_offset_ned'),
            'off_ground_threshold_m': LaunchConfiguration('off_ground_threshold_m'),
            'takeoff_timeout_s': LaunchConfiguration('takeoff_timeout_s'),
            'approach_first_timeout_s': LaunchConfiguration('approach_first_timeout_s'),
            'return_timeout_s': LaunchConfiguration('return_timeout_s'),
            'start_velocity_threshold_m_s': LaunchConfiguration('start_velocity_threshold_m_s'),
            'start_position_threshold_m': LaunchConfiguration('start_position_threshold_m'),
            'first_point_position_threshold_m': LaunchConfiguration(
                'first_point_position_threshold_m'),
            'first_point_velocity_threshold_m_s': LaunchConfiguration(
                'first_point_velocity_threshold_m_s'),
            'time_scale': LaunchConfiguration('time_scale'),
            'viz_world_frame': LaunchConfiguration('viz_world_frame'),
            'viz_waypoints_enu': LaunchConfiguration('viz_waypoints_enu'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )

    return LaunchDescription([
        play_mode,
        csv_path,
        waypoints_enu,
        publish_rate_hz,
        arm_wait_setpoints,
        hold_after_traj_s,
        hover_before_play_s,
        home_hover_altitude_m,
        land_after_done,
        landing_timeout_s,
        waypoint_timeout_s,
        use_acc_sp,
        origin_offset_ned,
        off_ground_threshold_m,
        takeoff_timeout_s,
        approach_first_timeout_s,
        return_timeout_s,
        start_velocity_threshold_m_s,
        start_position_threshold_m,
        first_point_position_threshold_m,
        first_point_velocity_threshold_m_s,
        time_scale,
        viz_world_frame,
        viz_waypoints_enu,
        use_sim_time,
        log_level,
        player_node,
    ])
