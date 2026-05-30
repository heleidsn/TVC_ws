#!/usr/bin/env python3
"""
TVC mission player (trajectory CSV or PX4 waypoint sequence).

``play_mode=trajectory`` (default): OFFBOARD CSV playback via ``TrajectorySetpoint``.
``play_mode=waypoint``: sequential ENU waypoints via ``GotoSetpoint`` in **OFFBOARD**
    (works with local/vision position; does not require GPS / global position).

Pipeline (trajectory)::

    [CSV file (ENU or NED)]
        |
        v
    [tvc_traj_player]  --(/fmu/in/trajectory_setpoint, NED)-->  PX4
        |
        +--(/fmu/in/offboard_control_mode heartbeat)--> PX4
        |
        +--(/fmu/in/vehicle_command: ARM, OFFBOARD)--> PX4

Pipeline (waypoint)::

    [tvc_traj_player]  --(/fmu/in/offboard_control_mode heartbeat)--> PX4
        |
        +--(/fmu/in/goto_setpoint, NED)--> PX4 GotoControl
        |
        +--(/fmu/in/vehicle_command: ARM, OFFBOARD, LAND)--> PX4

CSV format (produced by tvc_traj_opt_gui):
    Header lines starting with ``#`` (one of them contains ``frame: ENU`` or
    ``frame: NED``), followed by a column-name row, then numeric rows. Required
    columns: ``t,x,y,z,vx,vy,vz,qw,qx,qy,qz,yaw_deg``.

State machine (fixed mission profile for every CSV):
    INIT        -> ground setpoint at local home, then OFFBOARD + ARM
    ARMING      -> repeat ARM/OFFBOARD until accepted
    TAKEOFF     -> climb to home hover (0, 0, 1) ENU == (0, 0, -1) NED (+ offset)
    GOTO_FIRST  -> fly to CSV[0] (trajectory first point)
    HOVER_PRE   -> hold at CSV[0] for ``hover_before_play_s`` (default 2 s) once
                   |Δp| < ``first_point_position_threshold_m`` and
                   |v| < ``first_point_velocity_threshold_m_s`` (default 0.1 each)
    PLAYING     -> interpolate CSV as-is
    HOLDING     -> hold final CSV pose for ``hold_after_traj_s`` (default 2 s)
    RETURN      -> fly back to home hover (0, 0, 1), then LAND
    LANDING     -> wait for disarm / timeout
    DONE        -> exit

The publisher rate is configurable (``--rate``, default 50 Hz). The heartbeat
must be sent at >=2 Hz for PX4 to stay in OFFBOARD mode; we send it every
control tick.

RViz visualisation
==================
The node also publishes a few standard messages in the **world (ENU)** frame so
the planned trajectory can be inspected in RViz alongside the live PX4 odometry
(handled by ``px4_rviz_bridge``):

* ``/tvc_traj_player/planned_path``   (``nav_msgs/Path``, latched once at start)
* ``/tvc_traj_player/current_setpoint`` (``geometry_msgs/PoseStamped``, per tick)
* ``/tvc_traj_player/executed_path``  (``nav_msgs/Path``, grows over time)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from px4_msgs.msg import (GotoSetpoint, OffboardControlMode, TrajectorySetpoint,
                          VehicleCommand, VehicleLocalPosition, VehicleStatus)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


@dataclass
class TrajectorySample:
    """A single state sample interpolated from the CSV."""
    t: float
    pos_ned: np.ndarray          # (3,) [N, E, D]
    vel_ned: np.ndarray          # (3,) [vN, vE, vD]
    acc_ned: np.ndarray          # (3,) finite-difference; NaN for endpoints
    yaw_rad: float               # NED yaw (heading from North, clockwise)
    yawspeed_rad: float          # rad/s


def _load_csv(path: str) -> Tuple[np.ndarray, dict]:
    """Load a tvc trajectory CSV, returning (rows, meta).

    Returns
    -------
    rows : np.ndarray, shape (N, n_cols)
        Numeric rows.
    meta : dict
        ``columns`` (list[str]), ``frame`` ('ENU'/'NED'), ``method`` (str).
    """
    columns: Optional[List[str]] = None
    frame = 'ENU'  # default
    method = ''
    data_rows: List[List[float]] = []

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                low = line.lower()
                if 'frame:' in low:
                    if 'ned' in low:
                        frame = 'NED'
                    elif 'enu' in low:
                        frame = 'ENU'
                if 'method:' in low:
                    method = line.split(':', 1)[1].strip()
                continue
            if columns is None:
                columns = [c.strip() for c in line.split(',')]
                continue
            parts = [p.strip() for p in line.split(',')]
            row = [float('nan') if p.lower() == 'nan' else float(p) for p in parts]
            data_rows.append(row)

    if columns is None or not data_rows:
        raise ValueError(f'No numeric data found in {path}')

    rows = np.asarray(data_rows, dtype=float)
    if rows.shape[1] != len(columns):
        raise ValueError(
            f'Column count mismatch: header has {len(columns)}, data has {rows.shape[1]}'
        )
    return rows, {'columns': columns, 'frame': frame, 'method': method}


def _convert_to_ned(rows: np.ndarray, columns: List[str], frame: str) -> np.ndarray:
    """Convert the per-node state to NED in-place (returns a new array).

    For ENU input we apply ``(x,y,z) -> (y, x, -z)`` for position/velocity and
    convert yaw from ENU (CCW from East) to NED (CW from North).
    NED input is returned unchanged.
    """
    out = rows.copy()
    idx = {c: i for i, c in enumerate(columns)}

    def _col(name: str) -> int:
        if name not in idx:
            raise KeyError(f'CSV is missing required column: {name}')
        return idx[name]

    if frame.upper() == 'NED':
        return out

    # Position: x_NED = y_ENU, y_NED = x_ENU, z_NED = -z_ENU
    x_e = out[:, _col('x')].copy()
    y_e = out[:, _col('y')].copy()
    z_e = out[:, _col('z')].copy()
    out[:, _col('x')] = y_e
    out[:, _col('y')] = x_e
    out[:, _col('z')] = -z_e

    # Velocity (same rotation)
    vx_e = out[:, _col('vx')].copy()
    vy_e = out[:, _col('vy')].copy()
    vz_e = out[:, _col('vz')].copy()
    out[:, _col('vx')] = vy_e
    out[:, _col('vy')] = vx_e
    out[:, _col('vz')] = -vz_e

    # Yaw: ENU yaw is measured CCW from East. PX4 (NED) yaw is CW from North.
    # Relationship: yaw_NED = pi/2 - yaw_ENU, wrapped to (-pi, pi].
    if 'yaw_deg' in idx:
        yaw_enu_rad = np.deg2rad(out[:, _col('yaw_deg')])
        yaw_ned_rad = math.pi / 2.0 - yaw_enu_rad
        yaw_ned_rad = (yaw_ned_rad + math.pi) % (2.0 * math.pi) - math.pi
        out[:, _col('yaw_deg')] = np.rad2deg(yaw_ned_rad)

    return out


DEFAULT_WAYPOINT_HOVER_S = 2.0

DEFAULT_WAYPOINTS_ENU: List[List[float]] = [
    [0.0, 0.0, 1.5, 0.0, DEFAULT_WAYPOINT_HOVER_S],
    [1.0, 0.0, 1.5, 0.0, DEFAULT_WAYPOINT_HOVER_S],
    [1.0, 1.0, 1.5, 0.0, DEFAULT_WAYPOINT_HOVER_S],
    [-1.0, 1.0, 2.5, 0.0, DEFAULT_WAYPOINT_HOVER_S],
    [0.0, 1.0, 2.5, 0.0, DEFAULT_WAYPOINT_HOVER_S],
    [0.0, 0.0, 1.5, 0.0, DEFAULT_WAYPOINT_HOVER_S],
]


@dataclass
class WaypointNED:
    pos_ned: np.ndarray
    yaw_rad: float
    hover_s: float = DEFAULT_WAYPOINT_HOVER_S


def _parse_waypoints_enu(raw: object) -> List[List[float]]:
    if raw is None or raw == '':
        return [list(w) for w in DEFAULT_WAYPOINTS_ENU]
    if isinstance(raw, str):
        rows: List[List[float]] = []
        for part in raw.split(';'):
            part = part.strip()
            if not part:
                continue
            rows.append([float(x) for x in part.split(',')])
        return rows if rows else [list(w) for w in DEFAULT_WAYPOINTS_ENU]
    try:
        rows = [[float(v) for v in item] for item in raw]
        return rows if rows else [list(w) for w in DEFAULT_WAYPOINTS_ENU]
    except Exception:
        return [list(w) for w in DEFAULT_WAYPOINTS_ENU]


def _auto_yaw_enu_deg(waypoints_enu: Sequence[Sequence[float]], index: int) -> float:
    if index < len(waypoints_enu) - 1:
        dx = float(waypoints_enu[index + 1][0]) - float(waypoints_enu[index][0])
        dy = float(waypoints_enu[index + 1][1]) - float(waypoints_enu[index][1])
        if abs(dx) + abs(dy) > 1e-6:
            return math.degrees(math.atan2(dy, dx))
    if index > 0:
        dx = float(waypoints_enu[index][0]) - float(waypoints_enu[index - 1][0])
        dy = float(waypoints_enu[index][1]) - float(waypoints_enu[index - 1][1])
        if abs(dx) + abs(dy) > 1e-6:
            return math.degrees(math.atan2(dy, dx))
    return 0.0


def _enu_to_ned_waypoints(
    waypoints_enu: Sequence[Sequence[float]],
    origin_offset_ned: np.ndarray,
) -> List[WaypointNED]:
    out: List[WaypointNED] = []
    for i, row in enumerate(waypoints_enu):
        x_e, y_e, z_e = float(row[0]), float(row[1]), float(row[2])
        yaw_deg = (
            float(row[3]) if len(row) >= 4 else _auto_yaw_enu_deg(waypoints_enu, i)
        )
        hover_s = (
            float(row[4]) if len(row) >= 5 else DEFAULT_WAYPOINT_HOVER_S
        )
        pos_ned = origin_offset_ned + np.array([y_e, x_e, -z_e])
        yaw_enu_rad = math.radians(yaw_deg)
        yaw_ned = math.pi / 2.0 - yaw_enu_rad
        yaw_ned = (yaw_ned + math.pi) % (2.0 * math.pi) - math.pi
        out.append(WaypointNED(pos_ned=pos_ned, yaw_rad=yaw_ned, hover_s=hover_s))
    return out


class TVCTrajectoryPlayer(Node):
    """ROS 2 node: CSV trajectory (OFFBOARD) or waypoint sequence (GotoSetpoint)."""

    STATE_INIT = 'INIT'
    STATE_ARMING = 'ARMING'
    STATE_TAKEOFF = 'TAKEOFF'
    STATE_GOTO_FIRST = 'GOTO_FIRST'
    STATE_HOVER_PRE = 'HOVER_PRE'
    STATE_PLAYING = 'PLAYING'
    STATE_HOLDING = 'HOLDING'
    STATE_RETURN = 'RETURN'
    STATE_WP_EXECUTING = 'WP_EXECUTING'
    STATE_WP_HOVER = 'WP_HOVER'
    STATE_LANDING = 'LANDING'
    STATE_DONE = 'DONE'

    def __init__(self, param_overrides: Optional[dict] = None) -> None:
        super().__init__('tvc_traj_player')

        # ---- Parameters ---------------------------------------------------
        self.declare_parameter('play_mode', 'trajectory')
        self.declare_parameter('csv_path', '')
        self.declare_parameter('waypoints_enu', '')
        self.declare_parameter('waypoint_timeout_s', 120.0)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('arm_wait_setpoints', 20)
        self.declare_parameter('hold_after_traj_s', 2.0)
        self.declare_parameter('land_after_done', True)
        self.declare_parameter('landing_timeout_s', 30.0)
        self.declare_parameter('use_acceleration_setpoint', True)
        self.declare_parameter('home_hover_altitude_m', 1.0)
        self.declare_parameter('hover_before_play_s', 2.0)
        self.declare_parameter('off_ground_threshold_m', 0.3)
        self.declare_parameter('takeoff_timeout_s', 30.0)
        self.declare_parameter('approach_first_timeout_s', 60.0)
        self.declare_parameter('return_timeout_s', 60.0)
        self.declare_parameter('start_velocity_threshold_m_s', 0.1)
        self.declare_parameter('start_position_threshold_m', 0.5)
        # Stricter gate before CSV playback starts (GOTO_FIRST / HOVER_PRE only).
        self.declare_parameter('first_point_position_threshold_m', 0.1)
        self.declare_parameter('first_point_velocity_threshold_m_s', 0.1)
        # Time-domain scaling of the CSV trajectory. ``time_scale=2.0`` makes
        # the whole playback twice as long; commanded velocities are divided
        # by ``time_scale`` and accelerations by ``time_scale**2`` so the
        # geometric path is preserved but executed more slowly.
        self.declare_parameter('time_scale', 1.0)
        # Optional initial-position offset (NED, meters). Useful if the
        # autopilot's home is not exactly at the CSV origin.
        self.declare_parameter('origin_offset_ned', [0.0, 0.0, 0.0])
        # RViz visualisation frame (ENU). Must match px4_rviz_bridge / RViz.
        self.declare_parameter('viz_world_frame', 'world')

        overrides = param_overrides or {}
        self.play_mode = str(
            overrides.get('play_mode', self.get_parameter('play_mode').value)
        ).strip().lower()
        if self.play_mode not in ('trajectory', 'waypoint'):
            self.get_logger().warn(
                f'Unknown play_mode="{self.play_mode}"; using trajectory.'
            )
            self.play_mode = 'trajectory'
        self.get_logger().info(f'play_mode={self.play_mode}')
        csv_path = overrides.get('csv_path') or self.get_parameter(
            'csv_path'
        ).get_parameter_value().string_value
        self.rate_hz = float(overrides.get(
            'publish_rate_hz', self.get_parameter('publish_rate_hz').value
        ))
        self.arm_wait_setpoints = int(overrides.get(
            'arm_wait_setpoints', self.get_parameter('arm_wait_setpoints').value
        ))
        self.hold_after_traj_s = float(overrides.get(
            'hold_after_traj_s', self.get_parameter('hold_after_traj_s').value
        ))
        self.land_after_done = bool(overrides.get(
            'land_after_done', self.get_parameter('land_after_done').value
        ))
        self.landing_timeout_s = float(overrides.get(
            'landing_timeout_s', self.get_parameter('landing_timeout_s').value
        ))
        self.use_acceleration_setpoint = bool(overrides.get(
            'use_acceleration_setpoint',
            self.get_parameter('use_acceleration_setpoint').value,
        ))
        self.home_hover_altitude_m = float(overrides.get(
            'home_hover_altitude_m',
            self.get_parameter('home_hover_altitude_m').value,
        ))
        self.hover_before_play_s = float(overrides.get(
            'hover_before_play_s',
            self.get_parameter('hover_before_play_s').value,
        ))
        self.off_ground_threshold_m = float(overrides.get(
            'off_ground_threshold_m',
            self.get_parameter('off_ground_threshold_m').value,
        ))
        self.takeoff_timeout_s = float(overrides.get(
            'takeoff_timeout_s',
            self.get_parameter('takeoff_timeout_s').value,
        ))
        self.approach_first_timeout_s = float(overrides.get(
            'approach_first_timeout_s',
            self.get_parameter('approach_first_timeout_s').value,
        ))
        self.return_timeout_s = float(overrides.get(
            'return_timeout_s',
            self.get_parameter('return_timeout_s').value,
        ))
        self.start_velocity_threshold_m_s = float(overrides.get(
            'start_velocity_threshold_m_s',
            self.get_parameter('start_velocity_threshold_m_s').value,
        ))
        self.start_position_threshold_m = float(overrides.get(
            'start_position_threshold_m',
            self.get_parameter('start_position_threshold_m').value,
        ))
        self.first_point_position_threshold_m = float(overrides.get(
            'first_point_position_threshold_m',
            self.get_parameter('first_point_position_threshold_m').value,
        ))
        self.first_point_velocity_threshold_m_s = float(overrides.get(
            'first_point_velocity_threshold_m_s',
            self.get_parameter('first_point_velocity_threshold_m_s').value,
        ))
        self.waypoint_timeout_s = float(overrides.get(
            'waypoint_timeout_s',
            self.get_parameter('waypoint_timeout_s').value,
        ))
        self.time_scale = float(overrides.get(
            'time_scale', self.get_parameter('time_scale').value
        ))
        if self.time_scale <= 0.0:
            self.get_logger().warn(
                f'time_scale={self.time_scale} is non-positive; clamping to 1.0.'
            )
            self.time_scale = 1.0
        self.viz_world_frame = str(overrides.get(
            'viz_world_frame', self.get_parameter('viz_world_frame').value
        ))
        offset_param = self.get_parameter('origin_offset_ned').value
        try:
            self.origin_offset_ned = np.asarray(list(offset_param), dtype=float).reshape(3)
        except Exception:
            self.origin_offset_ned = np.zeros(3)

        self._waypoints: List[WaypointNED] = []
        if self.play_mode == 'waypoint':
            wp_raw = overrides.get(
                'waypoints_enu', self.get_parameter('waypoints_enu').value
            )
            wp_enu = _parse_waypoints_enu(wp_raw)
            if len(wp_enu) < 1:
                self.get_logger().error('At least one waypoint is required.')
                raise SystemExit(2)
            self._waypoints = _enu_to_ned_waypoints(wp_enu, self.origin_offset_ned)
        else:
            if not csv_path:
                self.get_logger().error(
                    'csv_path parameter is required (trajectory mode).'
                )
                raise SystemExit(2)
            if not os.path.isfile(csv_path):
                self.get_logger().error(f'CSV not found: {csv_path}')
                raise SystemExit(2)

            rows, meta = _load_csv(csv_path)
            frame = meta['frame']
            method = meta['method']
            cols = meta['columns']
            self.get_logger().info(
                f'Loaded {rows.shape[0]} samples from {csv_path} '
                f'(frame={frame}, method="{method}")'
            )
            rows_ned = _convert_to_ned(rows, cols, frame)
            idx = {c: i for i, c in enumerate(cols)}
            self._t = rows_ned[:, idx['t']]
            if self._t[0] != 0.0:
                self._t = self._t - self._t[0]
            self._pos = rows_ned[:, [idx['x'], idx['y'], idx['z']]]
            self._vel = rows_ned[:, [idx['vx'], idx['vy'], idx['vz']]]
            self._pos += self.origin_offset_ned[None, :]
            self._yaw = (
                np.deg2rad(rows_ned[:, idx['yaw_deg']])
                if 'yaw_deg' in idx else np.zeros(len(self._t))
            )
            self._acc = self._finite_diff(self._vel, self._t)
            self._yaw_rate = self._finite_diff_1d(self._yaw, self._t, wrap_angle=True)
            self.csv_duration = float(self._t[-1])
            self.traj_duration = self.csv_duration * self.time_scale

        # ---- QoS / publishers / subscribers ------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub_setpoint = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos
        )
        self.pub_offboard = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos
        )
        self.pub_goto = self.create_publisher(
            GotoSetpoint, '/fmu/in/goto_setpoint', qos
        )
        self.pub_cmd = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos
        )
        self.sub_status = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1',
            self._on_status, qos
        )
        self.sub_local_pos = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self._on_local_pos, qos
        )

        # ---- RViz visualisation publishers -------------------------------
        # Standard ROS QoS: reliable + transient_local for the latched planned
        # path; reliable + volatile for the live setpoint/executed path.
        viz_latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        viz_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub_planned_path = self.create_publisher(
            Path, '/tvc_traj_player/planned_path', viz_latched_qos
        )
        self.pub_current_setpoint = self.create_publisher(
            PoseStamped, '/tvc_traj_player/current_setpoint', viz_qos
        )
        self.pub_executed_path = self.create_publisher(
            Path, '/tvc_traj_player/executed_path', viz_qos
        )

        # ---- State -------------------------------------------------------
        self.state = self.STATE_INIT
        self.tick = 0
        self.vehicle_status: Optional[VehicleStatus] = None
        self.local_pos: Optional[VehicleLocalPosition] = None
        self.play_start_wall_ns: Optional[int] = None
        self.hold_start_wall_ns: Optional[int] = None
        self.hover_pre_start_wall_ns: Optional[int] = None
        # Takeoff / approach / return bookkeeping
        self.takeoff_start_wall_ns: Optional[int] = None
        self.goto_first_start_wall_ns: Optional[int] = None
        self.return_start_wall_ns: Optional[int] = None
        self.wp_index = 0
        self.wp_start_wall_ns: Optional[int] = None
        self.wp_hover_start_wall_ns: Optional[int] = None
        # Landing bookkeeping
        self.land_start_wall_ns: Optional[int] = None
        self._shutdown_requested: bool = False
        # Fixed home hover: ENU (0, 0, alt) -> NED (0, 0, -alt) + offset.
        self._home_hover_ned = self.origin_offset_ned + np.array(
            [0.0, 0.0, -self.home_hover_altitude_m]
        )
        self._ground_ned = self.origin_offset_ned.copy()
        self._home_yaw_rad = (
            float(self._yaw[0]) if self.play_mode == 'trajectory' else float(self._waypoints[0].yaw_rad)
        )
        # Growing buffer of executed setpoints (ENU) for RViz.
        self._executed_poses: List[PoseStamped] = []

        period = 1.0 / max(self.rate_hz, 1e-3)
        self.timer = self.create_timer(period, self._on_tick)

        self._publish_planned_path()

        if self.play_mode == 'waypoint':
            self.get_logger().info(
                f'Waypoint mode: {len(self._waypoints)} waypoints, '
                f'rate={self.rate_hz} Hz, pos_thr={self.start_position_threshold_m:.2f} m, '
                f'vel_thr={self.start_velocity_threshold_m_s:.2f} m/s'
            )
            for i, wp in enumerate(self._waypoints):
                self.get_logger().info(
                    f'  WP{i}: NED p={wp.pos_ned}, yaw={math.degrees(wp.yaw_rad):.1f} deg, '
                    f'hover={wp.hover_s:.1f}s'
                )
        else:
            self.get_logger().info(
                f'Trajectory mode: CSV duration={self.csv_duration:.2f}s, '
                f'time_scale={self.time_scale:.2f} -> playback={self.traj_duration:.2f}s, '
                f'rate={self.rate_hz} Hz, samples={len(self._t)}, '
                f'home_hover_altitude_m={self.home_hover_altitude_m:.2f}, '
                f'home_hover(NED)={self._home_hover_ned}, '
                f'hover_before_play_s={self.hover_before_play_s:.1f}, '
                f'hold_after_traj_s={self.hold_after_traj_s:.1f}, '
                f'start_velocity_threshold={self.start_velocity_threshold_m_s:.2f} m/s, '
                f'start_position_threshold={self.start_position_threshold_m:.2f} m, '
                f'first_point_thresholds: |Δp|<{self.first_point_position_threshold_m:.2f} m, '
                f'|v|<{self.first_point_velocity_threshold_m_s:.2f} m/s, '
                f'first_csv_setpoint(NED)=p={self._pos[0]}, '
                f'yaw={math.degrees(self._yaw[0]):.1f} deg'
            )

    # ---------------------------------------------------------------------
    # CSV helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _finite_diff(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Centered finite difference; endpoints use forward/backward differences."""
        n = len(t)
        out = np.zeros_like(arr)
        if n < 2:
            return out
        out[1:-1] = (arr[2:] - arr[:-2]) / (t[2:] - t[:-2])[:, None]
        out[0] = (arr[1] - arr[0]) / (t[1] - t[0])
        out[-1] = (arr[-1] - arr[-2]) / (t[-1] - t[-2])
        return out

    @staticmethod
    def _finite_diff_1d(arr: np.ndarray, t: np.ndarray, wrap_angle: bool = False) -> np.ndarray:
        n = len(t)
        out = np.zeros_like(arr)
        if n < 2:
            return out

        def _delta(a, b):
            d = b - a
            if wrap_angle:
                d = (d + math.pi) % (2.0 * math.pi) - math.pi
            return d

        for i in range(1, n - 1):
            out[i] = _delta(arr[i - 1], arr[i + 1]) / (t[i + 1] - t[i - 1])
        out[0] = _delta(arr[0], arr[1]) / (t[1] - t[0])
        out[-1] = _delta(arr[-2], arr[-1]) / (t[-1] - t[-2])
        return out

    def _interp(self, t_query: float) -> TrajectorySample:
        """Linear interpolation of the trajectory at physical time ``t_query``."""
        t_clamped = float(np.clip(t_query, self._t[0], self._t[-1]))
        i = int(np.searchsorted(self._t, t_clamped, side='right')) - 1
        i = max(0, min(i, len(self._t) - 2))
        t0, t1 = self._t[i], self._t[i + 1]
        alpha = 0.0 if t1 == t0 else (t_clamped - t0) / (t1 - t0)

        pos = self._pos[i] + alpha * (self._pos[i + 1] - self._pos[i])
        vel = self._vel[i] + alpha * (self._vel[i + 1] - self._vel[i])
        acc = self._acc[i] + alpha * (self._acc[i + 1] - self._acc[i])

        yaw0, yaw1 = self._yaw[i], self._yaw[i + 1]
        d_yaw = (yaw1 - yaw0 + math.pi) % (2.0 * math.pi) - math.pi
        yaw = yaw0 + alpha * d_yaw
        yawspeed = self._yaw_rate[i] + alpha * (self._yaw_rate[i + 1] - self._yaw_rate[i])
        return TrajectorySample(t_clamped, pos, vel, acc, yaw, yawspeed)

    # ---------------------------------------------------------------------
    # ROS callbacks
    # ---------------------------------------------------------------------
    def _on_status(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def _on_local_pos(self, msg: VehicleLocalPosition) -> None:
        self.local_pos = msg

    # ---------------------------------------------------------------------
    # PX4 helpers
    # ---------------------------------------------------------------------
    def _publish_heartbeat(self) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = bool(self.use_acceleration_setpoint)
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_offboard.publish(msg)

    def _publish_setpoint(self, s: TrajectorySample) -> None:
        msg = TrajectorySetpoint()
        msg.position = [float(s.pos_ned[0]), float(s.pos_ned[1]), float(s.pos_ned[2])]
        msg.velocity = [float(s.vel_ned[0]), float(s.vel_ned[1]), float(s.vel_ned[2])]
        if self.use_acceleration_setpoint and np.all(np.isfinite(s.acc_ned)):
            msg.acceleration = [float(s.acc_ned[0]), float(s.acc_ned[1]), float(s.acc_ned[2])]
        else:
            msg.acceleration = [float('nan')] * 3
        msg.jerk = [float('nan')] * 3
        msg.yaw = float(s.yaw_rad)
        msg.yawspeed = float(s.yawspeed_rad) if np.isfinite(s.yawspeed_rad) else 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_setpoint.publish(msg)

    def _publish_vehicle_command(self, command: int, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(params.get('param1', 0.0))
        msg.param2 = float(params.get('param2', 0.0))
        msg.param3 = float(params.get('param3', 0.0))
        msg.param4 = float(params.get('param4', 0.0))
        msg.param5 = float(params.get('param5', 0.0))
        msg.param6 = float(params.get('param6', 0.0))
        msg.param7 = float(params.get('param7', 0.0))
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_cmd.publish(msg)

    def _engage_offboard(self) -> None:
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
        )

    def _engage_auto_loiter(self) -> None:
        """AUTO LOITER requires a valid global position (GPS); not used in SITL+vision."""
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0, param2=4.0, param3=3.0,
        )

    def _arm(self) -> None:
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
        )

    def _disarm(self) -> None:
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0
        )

    def _land(self) -> None:
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    # ---------------------------------------------------------------------
    # RViz / visualisation helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _ned_to_enu_pos(p_ned: np.ndarray) -> Tuple[float, float, float]:
        """Convert a NED position triple to ENU (x_E=y_N, y_E=x_N, z_E=-z_D)."""
        return (float(p_ned[1]), float(p_ned[0]), float(-p_ned[2]))

    @staticmethod
    def _ned_yaw_to_enu_quat(yaw_ned_rad: float) -> Tuple[float, float, float, float]:
        """Convert a NED yaw (CW from North) to an ENU world-frame quaternion.

        Returns (x, y, z, w) – the ROS convention for ``geometry_msgs/Quaternion``.
        """
        yaw_enu = math.pi / 2.0 - yaw_ned_rad
        return (0.0, 0.0, math.sin(yaw_enu / 2.0), math.cos(yaw_enu / 2.0))

    def _make_pose_stamped(
        self, pos_ned: np.ndarray, yaw_ned_rad: float
    ) -> PoseStamped:
        x, y, z = self._ned_to_enu_pos(pos_ned)
        qx, qy, qz, qw = self._ned_yaw_to_enu_quat(yaw_ned_rad)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.viz_world_frame
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def _publish_planned_path(self) -> None:
        """Publish planned path (CSV or waypoints) as ``nav_msgs/Path`` (ENU)."""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.viz_world_frame
        if self.play_mode == 'waypoint':
            for wp in self._waypoints:
                pose = self._make_pose_stamped(wp.pos_ned, wp.yaw_rad)
                pose.header.frame_id = self.viz_world_frame
                path.poses.append(pose)
        else:
            for i in range(len(self._t)):
                pose = self._make_pose_stamped(self._pos[i], float(self._yaw[i]))
                pose.header.frame_id = self.viz_world_frame
                path.poses.append(pose)
        self.pub_planned_path.publish(path)
        self.get_logger().info(
            f'Published planned path on /tvc_traj_player/planned_path '
            f'(mode={self.play_mode}, n={len(path.poses)})'
        )

    def _publish_goto(self, wp: WaypointNED) -> None:
        msg = GotoSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float(wp.pos_ned[0]), float(wp.pos_ned[1]), float(wp.pos_ned[2])]
        msg.flag_control_heading = True
        msg.heading = float(wp.yaw_rad)
        msg.flag_set_max_horizontal_speed = False
        msg.max_horizontal_speed = 0.0
        msg.flag_set_max_vertical_speed = False
        msg.max_vertical_speed = 0.0
        msg.flag_set_max_heading_rate = False
        msg.max_heading_rate = 0.0
        self.pub_goto.publish(msg)
        pose = self._make_pose_stamped(wp.pos_ned, wp.yaw_rad)
        self.pub_current_setpoint.publish(pose)

    def _publish_current_setpoint(self, s: TrajectorySample) -> None:
        pose = self._make_pose_stamped(s.pos_ned, s.yaw_rad)
        self.pub_current_setpoint.publish(pose)

        # Throttled executed-path: keep one pose per ~5 ticks to bound size.
        if self.tick % 5 == 0:
            self._executed_poses.append(pose)
            if len(self._executed_poses) > 2000:
                self._executed_poses = self._executed_poses[-2000:]
            path = Path()
            path.header.stamp = pose.header.stamp
            path.header.frame_id = self.viz_world_frame
            path.poses = list(self._executed_poses)
            self.pub_executed_path.publish(path)

    # ---------------------------------------------------------------------
    # Tick state machine
    # ---------------------------------------------------------------------
    def _on_tick(self) -> None:
        self.tick += 1
        if self.play_mode == 'waypoint':
            self._on_tick_waypoint()
        else:
            self._on_tick_trajectory()

    def _on_tick_trajectory(self) -> None:
        self._publish_heartbeat()  # required every cycle while in OFFBOARD

        if self.state in (self.STATE_LANDING, self.STATE_DONE):
            self._on_tick_landing_done()
            return

        if self.state == self.STATE_INIT:
            # Hold a ground-level hover pose so PX4 has a valid setpoint to
            # latch onto before we ask for OFFBOARD/ARM.
            ground = self._initial_setpoint_sample()
            self._publish_setpoint(ground)
            self._publish_current_setpoint(ground)

            # Stream a few setpoints before requesting OFFBOARD/ARM (PX4 needs
            # to see a stable stream first).
            if self.tick == self.arm_wait_setpoints:
                self._engage_offboard()
                self._arm()
                self.state = self.STATE_ARMING
                self.get_logger().info(
                    f'Requested OFFBOARD + ARM after {self.tick} setpoints'
                )
            return

        if self.state == self.STATE_ARMING:
            ground = self._initial_setpoint_sample()
            self._publish_setpoint(ground)
            self._publish_current_setpoint(ground)
            if self.tick % int(self.rate_hz) == 0:
                # Re-issue the commands until PX4 acknowledges
                self._engage_offboard()
                self._arm()

            if self._is_armed_and_offboard():
                self._enter_pre_playback()
            return

        if self.state == self.STATE_TAKEOFF:
            sp = self._home_hover_sample()
            self._publish_setpoint(sp)
            self._publish_current_setpoint(sp)

            now_ns = self.get_clock().now().nanoseconds
            elapsed_s = (now_ns - self.takeoff_start_wall_ns) / 1e9
            pos_err = self._distance_to_target(sp.pos_ned)
            v_norm = self._velocity_norm()

            if self._is_airborne() and self.tick % max(1, int(self.rate_hz)) == 0:
                z_up = (-self.local_pos.z) if self.local_pos else float('nan')
                v_str = 'n/a' if v_norm is None else f'{v_norm:.3f}'
                self.get_logger().info(
                    f'TAKEOFF: z={z_up:.2f} m, |v|={v_str} m/s, '
                    f'|Δp|={pos_err:.2f} m, t={elapsed_s:.1f}s'
                )

            if self._ready_to_start_playback(sp.pos_ned):
                self.get_logger().info(
                    f'Reached home hover (0,0,{self.home_hover_altitude_m:.1f}) '
                    f'(|v|={v_norm:.3f} m/s, |Δp|={pos_err:.2f} m); '
                    f'flying to first trajectory point.'
                )
                self._enter_goto_first()
                return

            if elapsed_s >= self.takeoff_timeout_s:
                self.get_logger().warn(
                    f'Takeoff timeout ({self.takeoff_timeout_s:.1f}s) reached; '
                    f'proceeding to first trajectory point anyway '
                    f'(|v|={v_norm}, |Δp|={pos_err:.2f} m).'
                )
                self._enter_goto_first()
            return

        if self.state == self.STATE_GOTO_FIRST:
            sp = self._first_point_sample()
            self._publish_setpoint(sp)
            self._publish_current_setpoint(sp)

            now_ns = self.get_clock().now().nanoseconds
            elapsed_s = (now_ns - self.goto_first_start_wall_ns) / 1e9
            pos_err = self._distance_to_target(sp.pos_ned)
            v_norm = self._velocity_norm()

            if self._ready_at_first_point(sp.pos_ned):
                self.get_logger().info(
                    f'Reached first trajectory point '
                    f'(|v|={v_norm:.3f} m/s, |Δp|={pos_err:.2f} m); '
                    f'hovering {self.hover_before_play_s:.1f}s before playback.'
                )
                self._enter_hover_pre()
                return

            if self.tick % max(1, int(self.rate_hz)) == 0:
                v_str = 'n/a' if v_norm is None else f'{v_norm:.3f}'
                self.get_logger().info(
                    f'GOTO_FIRST: |v|={v_str} m/s, |Δp|={pos_err:.2f} m, t={elapsed_s:.1f}s '
                    f'(need |v|<{self.first_point_velocity_threshold_m_s:.2f}, '
                    f'|Δp|<{self.first_point_position_threshold_m:.2f})'
                )

            if elapsed_s >= self.approach_first_timeout_s:
                self.get_logger().warn(
                    f'Approach-first timeout ({self.approach_first_timeout_s:.1f}s) '
                    f'reached; still waiting for first point '
                    f'(|v|={v_norm}, |Δp|={pos_err:.2f} m).'
                )
            return

        if self.state == self.STATE_HOVER_PRE:
            sp = self._first_point_sample()
            self._publish_setpoint(sp)
            self._publish_current_setpoint(sp)

            now_ns = self.get_clock().now().nanoseconds
            v_norm = self._velocity_norm()
            pos_err = self._distance_to_target(sp.pos_ned)

            if not self._ready_at_first_point(sp.pos_ned):
                if self.hover_pre_start_wall_ns is not None:
                    self.get_logger().debug(
                        f'HOVER_PRE reset: |v|={v_norm}, |Δp|={pos_err:.2f} m'
                    )
                self.hover_pre_start_wall_ns = now_ns
                return

            held_s = (now_ns - self.hover_pre_start_wall_ns) / 1e9
            if held_s >= self.hover_before_play_s:
                self.get_logger().info(
                    f'Pre-play hover complete ({held_s:.1f}s, '
                    f'|v|={v_norm:.3f} m/s); starting trajectory.'
                )
                self._enter_playing()
            return

        if self.state == self.STATE_PLAYING:
            t_phys = (self.get_clock().now().nanoseconds - self.play_start_wall_ns) / 1e9
            if t_phys >= self.traj_duration:
                t_phys = self.traj_duration
                self.hold_start_wall_ns = self.get_clock().now().nanoseconds
                self.state = self.STATE_HOLDING
                self.get_logger().info(
                    f'Playback complete; holding final pose for {self.hold_after_traj_s:.1f}s'
                )
            sample = self._sample_scaled(t_phys)
            self._publish_setpoint(sample)
            self._publish_current_setpoint(sample)
            return

        if self.state == self.STATE_HOLDING:
            last = self._sample_scaled(self.traj_duration)
            last.vel_ned = np.zeros(3)
            last.acc_ned = np.zeros(3)
            last.yawspeed_rad = 0.0
            self._publish_setpoint(last)
            self._publish_current_setpoint(last)

            held_s = (self.get_clock().now().nanoseconds - self.hold_start_wall_ns) / 1e9
            if held_s >= self.hold_after_traj_s:
                if self.land_after_done:
                    self.get_logger().info(
                        f'Post-trajectory hold complete ({held_s:.1f}s); '
                        f'returning to home hover.'
                    )
                    self._enter_return()
                else:
                    self.get_logger().info(
                        'Hold complete; land_after_done=false, exiting.'
                    )
                    self.state = self.STATE_DONE
                    self._request_shutdown('hold complete (no land requested)')
            return

        if self.state == self.STATE_RETURN:
            sp = self._home_hover_sample()
            self._publish_setpoint(sp)
            self._publish_current_setpoint(sp)

            now_ns = self.get_clock().now().nanoseconds
            elapsed_s = (now_ns - self.return_start_wall_ns) / 1e9
            pos_err = self._distance_to_target(sp.pos_ned)
            v_norm = self._velocity_norm()

            if self._ready_to_start_playback(sp.pos_ned):
                self.get_logger().info(
                    f'Reached home hover (|v|={v_norm:.3f} m/s, |Δp|={pos_err:.2f} m); '
                    f'requesting LAND.'
                )
                self._land()
                self.state = self.STATE_LANDING
                self.land_start_wall_ns = now_ns
                return

            if self.tick % max(1, int(self.rate_hz)) == 0:
                v_str = 'n/a' if v_norm is None else f'{v_norm:.3f}'
                self.get_logger().info(
                    f'RETURN: |v|={v_str} m/s, |Δp|={pos_err:.2f} m, t={elapsed_s:.1f}s'
                )

            if elapsed_s >= self.return_timeout_s:
                self.get_logger().warn(
                    f'Return timeout ({self.return_timeout_s:.1f}s) reached; '
                    f'requesting LAND anyway (|Δp|={pos_err:.2f} m).'
                )
                self._land()
                self.state = self.STATE_LANDING
                self.land_start_wall_ns = now_ns
            return

    def _on_tick_waypoint(self) -> None:
        # OFFBOARD heartbeat is required; GotoSetpoint drives motion (no ext. trajectory).
        self._publish_heartbeat()

        if self.state in (self.STATE_LANDING, self.STATE_DONE):
            self._on_tick_landing_done()
            return

        if self.state == self.STATE_INIT:
            hold = self._first_point_sample()
            self._publish_setpoint(hold)
            self._publish_current_setpoint(hold)
            if self.tick >= self.arm_wait_setpoints:
                self._engage_offboard()
                self._arm()
                self.state = self.STATE_ARMING
                self.get_logger().info('Requested OFFBOARD + ARM (waypoint mode).')
            return

        if self.state == self.STATE_ARMING:
            hold = self._first_point_sample()
            self._publish_setpoint(hold)
            self._publish_current_setpoint(hold)
            if self.tick % max(1, int(self.rate_hz)) == 0:
                self._engage_offboard()
                self._arm()
            if self._is_armed_and_offboard():
                self.wp_index = 0
                self.wp_start_wall_ns = self.get_clock().now().nanoseconds
                self.state = self.STATE_WP_EXECUTING
                self.get_logger().info(
                    f'Armed + OFFBOARD; flying to waypoint 0/{len(self._waypoints) - 1}.'
                )
            return

        if self.state == self.STATE_WP_EXECUTING:
            wp = self._waypoints[self.wp_index]
            self._publish_goto(wp)
            now_ns = self.get_clock().now().nanoseconds
            elapsed_s = (now_ns - self.wp_start_wall_ns) / 1e9
            pos_err = self._distance_to_target(wp.pos_ned)
            v_norm = self._velocity_norm()

            if self._ready_to_start_playback(wp.pos_ned):
                self.get_logger().info(
                    f'Reached waypoint {self.wp_index} '
                    f'(|v|={v_norm:.3f} m/s, |Δp|={pos_err:.2f} m); '
                    f'hovering {wp.hover_s:.1f}s.'
                )
                self.wp_hover_start_wall_ns = now_ns
                self.state = self.STATE_WP_HOVER
                return

            if self.tick % max(1, int(self.rate_hz)) == 0:
                v_str = 'n/a' if v_norm is None else f'{v_norm:.3f}'
                self.get_logger().info(
                    f'WP{self.wp_index}: |v|={v_str} m/s, |Δp|={pos_err:.2f} m, '
                    f't={elapsed_s:.1f}s'
                )

            if elapsed_s >= self.waypoint_timeout_s:
                self.get_logger().warn(
                    f'Waypoint {self.wp_index} timeout ({self.waypoint_timeout_s:.1f}s); '
                    f'advancing (|Δp|={pos_err:.2f} m).'
                )
                self.wp_hover_start_wall_ns = now_ns
                self.state = self.STATE_WP_HOVER
            return

        if self.state == self.STATE_WP_HOVER:
            wp = self._waypoints[self.wp_index]
            self._publish_goto(wp)
            now_ns = self.get_clock().now().nanoseconds
            pos_err = self._distance_to_target(wp.pos_ned)
            v_norm = self._velocity_norm()

            if not self._ready_to_start_playback(wp.pos_ned):
                self.wp_hover_start_wall_ns = now_ns
                return

            held_s = (now_ns - self.wp_hover_start_wall_ns) / 1e9
            if held_s < wp.hover_s:
                return

            self.get_logger().info(
                f'Waypoint {self.wp_index} hover complete ({held_s:.1f}s); '
                f'|v|={v_norm:.3f} m/s, |Δp|={pos_err:.2f} m.'
            )
            if self.wp_index >= len(self._waypoints) - 1:
                if self.land_after_done:
                    self._land()
                    self.land_start_wall_ns = now_ns
                    self.state = self.STATE_LANDING
                    self.get_logger().info('All waypoints done; requesting LAND.')
                else:
                    self.state = self.STATE_DONE
                    self._request_shutdown('all waypoints reached')
                return

            self.wp_index += 1
            self.wp_start_wall_ns = now_ns
            self.state = self.STATE_WP_EXECUTING
            self.get_logger().info(
                f'Flying to waypoint {self.wp_index}/{len(self._waypoints) - 1}.'
            )
            return

    def _on_tick_landing_done(self) -> None:
        if self.state == self.STATE_LANDING:
            disarmed = (
                self.vehicle_status is not None
                and self.vehicle_status.arming_state
                == VehicleStatus.ARMING_STATE_DISARMED
            )
            if disarmed:
                self.state = self.STATE_DONE
                self.get_logger().info('Vehicle disarmed; landing complete.')
                self._request_shutdown('landing complete')
                return

            now_ns = self.get_clock().now().nanoseconds
            elapsed_s = (
                (now_ns - self.land_start_wall_ns) / 1e9
                if self.land_start_wall_ns is not None else 0.0
            )
            if self.tick % max(1, int(self.rate_hz)) == 0:
                z_up = (-self.local_pos.z) if self.local_pos else float('nan')
                self.get_logger().info(
                    f'LANDING in progress: t={elapsed_s:.1f}s, altitude={z_up:.2f} m'
                )
            if elapsed_s >= self.landing_timeout_s:
                self.get_logger().warn(
                    f'Landing timeout ({self.landing_timeout_s:.1f}s) reached; '
                    f'forcing DISARM and shutting down.'
                )
                self._disarm()
                self.state = self.STATE_DONE
                self._request_shutdown('landing timeout')
            return

        if self.state == self.STATE_DONE:
            self._request_shutdown('STATE_DONE reached')

    # ---------------------------------------------------------------------
    # State entry helpers
    # ---------------------------------------------------------------------
    def _initial_setpoint_sample(self) -> TrajectorySample:
        """Sample used during INIT/ARMING – hold at ground home (local origin)."""
        return TrajectorySample(
            t=0.0,
            pos_ned=self._ground_ned.copy(),
            vel_ned=np.zeros(3),
            acc_ned=np.zeros(3),
            yaw_rad=self._home_yaw_rad,
            yawspeed_rad=0.0,
        )

    def _home_hover_sample(self) -> TrajectorySample:
        """Hover setpoint at fixed home (0, 0, alt) in ENU / NED."""
        return TrajectorySample(
            t=0.0,
            pos_ned=self._home_hover_ned.copy(),
            vel_ned=np.zeros(3),
            acc_ned=np.zeros(3),
            yaw_rad=self._home_yaw_rad,
            yawspeed_rad=0.0,
        )

    def _first_point_sample(self) -> TrajectorySample:
        """Hover setpoint at CSV[0] or first ENU waypoint (trajectory / waypoint mode)."""
        if self.play_mode == 'waypoint':
            wp = self._waypoints[0]
            return TrajectorySample(
                t=0.0,
                pos_ned=wp.pos_ned.copy(),
                vel_ned=np.zeros(3),
                acc_ned=np.zeros(3),
                yaw_rad=wp.yaw_rad,
                yawspeed_rad=0.0,
            )
        return TrajectorySample(
            t=0.0,
            pos_ned=self._pos[0].copy(),
            vel_ned=np.zeros(3),
            acc_ned=np.zeros(3),
            yaw_rad=float(self._yaw[0]),
            yawspeed_rad=0.0,
        )

    def _enter_pre_playback(self) -> None:
        """Climb to home hover, then goto CSV[0], hover, then PLAYING."""
        self.takeoff_start_wall_ns = self.get_clock().now().nanoseconds
        self.state = self.STATE_TAKEOFF
        self.get_logger().info(
            f'Armed + OFFBOARD; taking off to home hover '
            f'(0, 0, {self.home_hover_altitude_m:.1f}) ENU, '
            f'NED p={self._home_hover_ned}.'
        )

    def _enter_goto_first(self) -> None:
        self.goto_first_start_wall_ns = self.get_clock().now().nanoseconds
        self.state = self.STATE_GOTO_FIRST
        first = self._pos[0]
        self.get_logger().info(
            f'Flying to first trajectory point (NED p={first}).'
        )

    def _enter_hover_pre(self) -> None:
        self.hover_pre_start_wall_ns = self.get_clock().now().nanoseconds
        self.state = self.STATE_HOVER_PRE

    def _enter_return(self) -> None:
        self.return_start_wall_ns = self.get_clock().now().nanoseconds
        self.state = self.STATE_RETURN

    def _enter_playing(self) -> None:
        self.play_start_wall_ns = self.get_clock().now().nanoseconds
        self.state = self.STATE_PLAYING
        self.get_logger().info(
            f'Starting trajectory playback ({self.traj_duration:.2f}s, '
            f'{len(self._t)} samples).'
        )

    def _request_shutdown(self, reason: str) -> None:
        """Cleanly stop the timer and ask rclpy to exit ``spin``.

        Safe to call multiple times. After this returns, ``main()`` will
        clean up the node and the process will exit with code 0.
        """
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self.get_logger().info(f'Shutting down tvc_traj_player: {reason}.')
        try:
            if self.timer is not None:
                self.timer.cancel()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def _is_airborne(self) -> bool:
        """True when the vehicle's altitude (above local origin) exceeds the
        configured off-ground threshold."""
        if self.local_pos is None:
            return False
        # In NED, "up" is the -z direction. ``z_valid`` indicates whether the
        # estimator is happy with the z estimate.
        if hasattr(self.local_pos, 'z_valid') and not self.local_pos.z_valid:
            return False
        altitude_up_m = -float(self.local_pos.z)
        return altitude_up_m >= self.off_ground_threshold_m

    def _velocity_norm(self) -> Optional[float]:
        """Return ``||v||`` from the latest VehicleLocalPosition, or None."""
        if self.local_pos is None:
            return None
        if hasattr(self.local_pos, 'v_xy_valid') and not self.local_pos.v_xy_valid:
            return None
        vx = float(self.local_pos.vx)
        vy = float(self.local_pos.vy)
        vz = float(self.local_pos.vz)
        return math.sqrt(vx * vx + vy * vy + vz * vz)

    def _distance_to_target(self, target_ned: np.ndarray) -> float:
        """Euclidean distance between the latest local position and a NED point."""
        if self.local_pos is None:
            return float('inf')
        if hasattr(self.local_pos, 'xy_valid') and not self.local_pos.xy_valid:
            return float('inf')
        if hasattr(self.local_pos, 'z_valid') and not self.local_pos.z_valid:
            return float('inf')
        dx = float(self.local_pos.x) - float(target_ned[0])
        dy = float(self.local_pos.y) - float(target_ned[1])
        dz = float(self.local_pos.z) - float(target_ned[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _ready_to_start_playback(self, target_ned: np.ndarray) -> bool:
        """True when the vehicle is close enough and slow enough at a target."""
        v_norm = self._velocity_norm()
        if v_norm is None:
            return False
        pos_err = self._distance_to_target(target_ned)
        if not math.isfinite(pos_err):
            return False
        return (
            pos_err <= self.start_position_threshold_m
            and v_norm < self.start_velocity_threshold_m_s
        )

    def _ready_at_first_point(self, target_ned: np.ndarray) -> bool:
        """Stricter arrival check at CSV[0] before trajectory playback."""
        v_norm = self._velocity_norm()
        if v_norm is None:
            return False
        pos_err = self._distance_to_target(target_ned)
        if not math.isfinite(pos_err):
            return False
        return (
            pos_err < self.first_point_position_threshold_m
            and v_norm < self.first_point_velocity_threshold_m_s
        )

    def _sample_scaled(self, t_phys: float) -> TrajectorySample:
        """Return a sample at physical time ``t_phys`` with time-scale applied.

        ``time_scale > 1`` slows the trajectory down: the geometric path is
        unchanged but velocities are divided by ``time_scale`` and
        accelerations by ``time_scale**2``.
        """
        s = max(self.time_scale, 1e-6)
        t_csv = t_phys / s
        sample = self._interp(t_csv)
        if s != 1.0:
            sample.vel_ned = sample.vel_ned / s
            sample.acc_ned = sample.acc_ned / (s * s)
            sample.yawspeed_rad = float(sample.yawspeed_rad) / s
        # Report the physical playback time, not the CSV time.
        sample.t = float(t_phys)
        return sample

    def _is_armed_and_offboard(self) -> bool:
        if self.vehicle_status is None:
            return False
        return (
            self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
            and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )


def _default_trajectory_csv() -> str:
    """Newest *.csv under TVC-traj-opt/trajs/ (install layout or source tree)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'TVC-traj-opt', 'trajs')),
        '/home/helei/TVC_ws/TVC-traj-opt/trajs',
    ]
    for directory in candidates:
        if not os.path.isdir(directory):
            continue
        csv_files = [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.endswith('.csv')
        ]
        if csv_files:
            return max(csv_files, key=os.path.getmtime)
    return ''


def main(args=None, param_overrides: Optional[dict] = None) -> int:
    # Strip ROS-specific argv; optional CLI overrides are passed via param_overrides.
    init_args = None
    if args is not None:
        if '--ros-args' in args:
            init_args = args[: args.index('--ros-args') + 1]
        else:
            init_args = [args[0]] if args else None
    rclpy.init(args=init_args)
    try:
        node = TVCTrajectoryPlayer(param_overrides=param_overrides)
    except SystemExit as e:
        if rclpy.ok():
            rclpy.shutdown()
        return int(e.code) if e.code is not None else 1

    exit_code = 0
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted; shutting down.')
        exit_code = 130
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == '__main__':
    # Allow: python3 tvc_traj_player.py /path/to/traj.csv [--rate 50] [--no-land]
    import argparse

    parser = argparse.ArgumentParser(
        description='TVC mission player: CSV trajectory or PX4 waypoint sequence.',
    )
    parser.add_argument(
        '--mode', choices=('trajectory', 'waypoint'), default=None,
        help='Play mode: trajectory (CSV offboard) or waypoint (GotoSetpoint).',
    )
    parser.add_argument(
        'csv_path',
        nargs='?',
        default=None,
        help='Trajectory CSV path (trajectory mode only).',
    )
    parser.add_argument('--rate', type=float, default=None, help='Publish rate [Hz]')
    parser.add_argument(
        '--time-scale', type=float, default=None,
        help='Time scaling factor (>1 slows the trajectory down).',
    )
    parser.add_argument(
        '--start-position-threshold', type=float, default=None,
        help='Max distance [m] to a hover target before advancing the mission.',
    )
    parser.add_argument(
        '--start-velocity-threshold', type=float, default=None,
        help='Speed [m/s] below which a hover target is considered reached.',
    )
    parser.add_argument(
        '--home-hover-altitude', type=float, default=None,
        help='Fixed home hover altitude (0,0,alt) ENU before/after trajectory [m].',
    )
    parser.add_argument(
        '--hover-before-play', type=float, default=None,
        help='Seconds to hover at home before starting the CSV [s].',
    )
    parser.add_argument(
        '--hold-after-traj', type=float, default=None,
        help='Seconds to hold the final CSV pose before returning home [s].',
    )
    parser.add_argument(
        '--no-land', action='store_true',
        help='Do not return home and land after trajectory ends',
    )
    cli_args = parser.parse_args()

    overrides: dict = {}
    if cli_args.mode is not None:
        overrides['play_mode'] = cli_args.mode

    if cli_args.mode != 'waypoint':
        csv_path = cli_args.csv_path or _default_trajectory_csv()
        if not csv_path:
            parser.error(
                'csv_path is required in trajectory mode: pass a file path or '
                'export a CSV to TVC-traj-opt/trajs/'
            )
        overrides['csv_path'] = csv_path
    if cli_args.rate is not None:
        overrides['publish_rate_hz'] = cli_args.rate
    if cli_args.time_scale is not None:
        overrides['time_scale'] = cli_args.time_scale
    if cli_args.start_velocity_threshold is not None:
        overrides['start_velocity_threshold_m_s'] = cli_args.start_velocity_threshold
    if cli_args.start_position_threshold is not None:
        overrides['start_position_threshold_m'] = cli_args.start_position_threshold
    if cli_args.home_hover_altitude is not None:
        overrides['home_hover_altitude_m'] = cli_args.home_hover_altitude
    if cli_args.hover_before_play is not None:
        overrides['hover_before_play_s'] = cli_args.hover_before_play
    if cli_args.hold_after_traj is not None:
        overrides['hold_after_traj_s'] = cli_args.hold_after_traj
    if cli_args.no_land:
        overrides['land_after_done'] = False

    sys.exit(main(param_overrides=overrides) or 0)
