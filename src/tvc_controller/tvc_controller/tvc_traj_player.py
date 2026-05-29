#!/usr/bin/env python3
"""
TVC trajectory player.

Loads a CSV file exported by ``tvc_traj_opt_gui`` and publishes the trajectory
as PX4 ``TrajectorySetpoint`` messages so PX4 (or any downstream controller)
can track it.

Pipeline::

    [CSV file (ENU or NED)]
        |
        v
    [tvc_traj_player]  --(/fmu/in/trajectory_setpoint, NED)-->  PX4
        |
        +--(/fmu/in/offboard_control_mode heartbeat)--> PX4
        |
        +--(/fmu/in/vehicle_command: ARM, OFFBOARD)--> PX4

CSV format (produced by tvc_traj_opt_gui):
    Header lines starting with ``#`` (one of them contains ``frame: ENU`` or
    ``frame: NED``), followed by a column-name row, then numeric rows. Required
    columns: ``t,x,y,z,vx,vy,vz,qw,qx,qy,qz,yaw_deg``.

State machine:
    INIT      -> send heartbeat + hover setpoint, wait for OFFBOARD + ARMED
    ARMING    -> request OFFBOARD then ARM repeatedly until accepted
    TAKEOFF   -> publish a hover setpoint at the first trajectory point
                 (CSV[0] shifted up by ``takeoff_altitude_m``); transition once
                 the vehicle is airborne and ``|v| < start_velocity_threshold_m_s``
                 (i.e. it has actually reached and settled at the first point)
    PLAYING   -> publish setpoints by interpolating the CSV. The mapping between
                 physical time and CSV time is ``t_csv = t_phys / time_scale``.
                 ``time_scale > 1`` slows the trajectory down: velocities are
                 divided by ``time_scale`` and accelerations by ``time_scale**2``
    HOLDING   -> CSV done: keep publishing the last sample for a short hold
    LANDING   -> request LAND mode, then DISARM, then exit

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
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
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


class TVCTrajectoryPlayer(Node):
    """ROS 2 node that streams a CSV trajectory to PX4 as TrajectorySetpoint."""

    STATE_INIT = 'INIT'
    STATE_ARMING = 'ARMING'
    STATE_TAKEOFF = 'TAKEOFF'
    STATE_PLAYING = 'PLAYING'
    STATE_HOLDING = 'HOLDING'
    STATE_LANDING = 'LANDING'
    STATE_DONE = 'DONE'

    def __init__(self, param_overrides: Optional[dict] = None) -> None:
        super().__init__('tvc_traj_player')

        # ---- Parameters ---------------------------------------------------
        self.declare_parameter('csv_path', '')
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('arm_wait_setpoints', 20)
        self.declare_parameter('hold_after_done_s', 2.0)
        self.declare_parameter('land_after_done', True)
        # Safety timeout for the LANDING stage. If PX4 has not switched to
        # DISARMED within this many seconds after the LAND command, force a
        # DISARM and shut the node down anyway.
        self.declare_parameter('landing_timeout_s', 30.0)
        self.declare_parameter('use_acceleration_setpoint', True)
        # Takeoff stage --------------------------------------------------
        # When True, after ARM/OFFBOARD we first hover at takeoff_altitude_m
        # and only start replaying the CSV once the vehicle leaves the ground.
        # The whole CSV trajectory is then shifted up by takeoff_altitude_m
        # so its z=0 origin lines up with the hover point.
        self.declare_parameter('do_takeoff', True)
        self.declare_parameter('takeoff_altitude_m', 1.0)
        self.declare_parameter('off_ground_threshold_m', 0.3)
        self.declare_parameter('takeoff_settle_time_s', 1.0)
        self.declare_parameter('takeoff_timeout_s', 30.0)
        # Velocity at which the vehicle is considered "settled" at the first
        # trajectory point – then trajectory playback starts. [m/s]
        self.declare_parameter('start_velocity_threshold_m_s', 0.1)
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
        csv_path = overrides.get('csv_path') or self.get_parameter(
            'csv_path'
        ).get_parameter_value().string_value
        self.rate_hz = float(overrides.get(
            'publish_rate_hz', self.get_parameter('publish_rate_hz').value
        ))
        self.arm_wait_setpoints = int(overrides.get(
            'arm_wait_setpoints', self.get_parameter('arm_wait_setpoints').value
        ))
        self.hold_after_done_s = float(overrides.get(
            'hold_after_done_s', self.get_parameter('hold_after_done_s').value
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
        self.do_takeoff = bool(overrides.get(
            'do_takeoff', self.get_parameter('do_takeoff').value
        ))
        self.takeoff_altitude_m = float(overrides.get(
            'takeoff_altitude_m', self.get_parameter('takeoff_altitude_m').value
        ))
        self.off_ground_threshold_m = float(overrides.get(
            'off_ground_threshold_m',
            self.get_parameter('off_ground_threshold_m').value,
        ))
        self.takeoff_settle_time_s = float(overrides.get(
            'takeoff_settle_time_s',
            self.get_parameter('takeoff_settle_time_s').value,
        ))
        self.takeoff_timeout_s = float(overrides.get(
            'takeoff_timeout_s',
            self.get_parameter('takeoff_timeout_s').value,
        ))
        self.start_velocity_threshold_m_s = float(overrides.get(
            'start_velocity_threshold_m_s',
            self.get_parameter('start_velocity_threshold_m_s').value,
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

        if not csv_path:
            self.get_logger().error(
                'csv_path parameter is required (path to a tvc trajectory CSV).'
            )
            raise SystemExit(2)
        if not os.path.isfile(csv_path):
            self.get_logger().error(f'CSV not found: {csv_path}')
            raise SystemExit(2)

        # ---- Load + convert CSV ------------------------------------------
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
        # When using a separate takeoff stage, shift the whole trajectory up
        # by takeoff_altitude_m so CSV t=0 aligns with the hover altitude.
        # (In NED, "up" is the -z direction, hence the subtraction.)
        if self.do_takeoff:
            self._pos[:, 2] -= self.takeoff_altitude_m
        self._yaw = np.deg2rad(rows_ned[:, idx['yaw_deg']]) if 'yaw_deg' in idx else np.zeros(len(self._t))
        self._acc = self._finite_diff(self._vel, self._t)
        self._yaw_rate = self._finite_diff_1d(self._yaw, self._t, wrap_angle=True)
        # csv_duration is the trajectory length expressed in CSV time;
        # traj_duration is its physical playback length (after time_scale).
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
        # Takeoff bookkeeping
        self.takeoff_start_wall_ns: Optional[int] = None
        self.off_ground_wall_ns: Optional[int] = None
        # Landing bookkeeping
        self.land_start_wall_ns: Optional[int] = None
        self._shutdown_requested: bool = False
        # NED hover target during the TAKEOFF stage; resolved when entering it.
        self._takeoff_hover_ned: Optional[np.ndarray] = None
        self._takeoff_yaw_rad: float = float(self._yaw[0])
        # Growing buffer of executed setpoints (ENU) for RViz.
        self._executed_poses: List[PoseStamped] = []

        period = 1.0 / max(self.rate_hz, 1e-3)
        self.timer = self.create_timer(period, self._on_tick)

        # Publish the (latched) planned path so RViz can render it immediately.
        self._publish_planned_path()

        self.get_logger().info(
            f'Player armed. CSV duration={self.csv_duration:.2f}s, '
            f'time_scale={self.time_scale:.2f} -> playback={self.traj_duration:.2f}s, '
            f'rate={self.rate_hz} Hz, samples={len(self._t)}, '
            f'do_takeoff={self.do_takeoff}, takeoff_altitude_m={self.takeoff_altitude_m:.2f}, '
            f'start_velocity_threshold={self.start_velocity_threshold_m_s:.2f} m/s, '
            f'first_setpoint(NED)=p={self._pos[0]}, yaw={math.degrees(self._yaw[0]):.1f} deg'
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
        """Publish the full CSV trajectory as a ``nav_msgs/Path`` (ENU)."""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.viz_world_frame
        for i in range(len(self._t)):
            pose = self._make_pose_stamped(self._pos[i], float(self._yaw[i]))
            pose.header.frame_id = self.viz_world_frame
            path.poses.append(pose)
        self.pub_planned_path.publish(path)
        self.get_logger().info(
            f'Published planned path on /tvc_traj_player/planned_path '
            f'(frame_id="{self.viz_world_frame}", n={len(path.poses)})'
        )

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
        self._publish_heartbeat()  # required every cycle while in OFFBOARD

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
                if self.do_takeoff:
                    self._enter_takeoff()
                else:
                    self._enter_playing()
            return

        if self.state == self.STATE_TAKEOFF:
            sp = self._takeoff_setpoint_sample()
            self._publish_setpoint(sp)
            self._publish_current_setpoint(sp)

            now_ns = self.get_clock().now().nanoseconds
            elapsed_s = (now_ns - self.takeoff_start_wall_ns) / 1e9

            airborne = self._is_airborne()
            if airborne and self.off_ground_wall_ns is None:
                self.off_ground_wall_ns = now_ns
                z_up = (-self.local_pos.z) if self.local_pos else float('nan')
                self.get_logger().info(
                    f'Off the ground at t={elapsed_s:.2f}s (z={z_up:.2f} m). '
                    f'Waiting for vehicle to settle at first trajectory point '
                    f'(|v| < {self.start_velocity_threshold_m_s:.2f} m/s).'
                )

            # Transition criterion: once we are off-ground AND have hovered
            # for at least takeoff_settle_time_s, wait for the velocity to
            # drop below the configured threshold (i.e. the vehicle has
            # reached and stabilised at the first trajectory point).
            if self.off_ground_wall_ns is not None:
                settled_s = (now_ns - self.off_ground_wall_ns) / 1e9
                v_norm = self._velocity_norm()
                pos_err = self._distance_to_target(sp.pos_ned)
                if (
                    settled_s >= self.takeoff_settle_time_s
                    and v_norm is not None
                    and v_norm < self.start_velocity_threshold_m_s
                ):
                    self.get_logger().info(
                        f'Settled at first trajectory point '
                        f'(|v|={v_norm:.3f} m/s, |Δp|={pos_err:.2f} m) '
                        f'after {settled_s:.2f}s; starting trajectory.'
                    )
                    self._enter_playing()
                    return

                # Periodic progress log (every ~1s) so the user can see why
                # we have not transitioned yet.
                if self.tick % max(1, int(self.rate_hz)) == 0:
                    v_str = 'n/a' if v_norm is None else f'{v_norm:.3f}'
                    self.get_logger().info(
                        f'TAKEOFF settling: |v|={v_str} m/s, '
                        f'|Δp|={pos_err:.2f} m, settled_s={settled_s:.2f}'
                    )

            # Safety: if takeoff has not been achieved within timeout, log and
            # fall back to playing the trajectory anyway.
            if elapsed_s >= self.takeoff_timeout_s:
                self.get_logger().warn(
                    f'Takeoff timeout ({self.takeoff_timeout_s:.1f}s) reached; '
                    f'starting trajectory playback even though velocity has '
                    f'not settled below threshold.'
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
                    f'Playback complete; holding final pose for {self.hold_after_done_s:.1f}s'
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
            if held_s >= self.hold_after_done_s:
                if self.land_after_done:
                    self._land()
                    self.state = self.STATE_LANDING
                    self.land_start_wall_ns = self.get_clock().now().nanoseconds
                    self.get_logger().info('Hold complete; requesting LAND.')
                else:
                    self.get_logger().info(
                        'Hold complete; land_after_done=false, exiting.'
                    )
                    self.state = self.STATE_DONE
                    self._request_shutdown('hold complete (no land requested)')
            return

        if self.state == self.STATE_LANDING:
            disarmed = (
                self.vehicle_status is not None
                and self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
            )
            if disarmed:
                self.state = self.STATE_DONE
                self.get_logger().info('Vehicle disarmed; landing complete.')
                self._request_shutdown('landing complete')
                return

            now_ns = self.get_clock().now().nanoseconds
            elapsed_s = (
                (now_ns - self.land_start_wall_ns) / 1e9
                if self.land_start_wall_ns is not None
                else 0.0
            )
            # Periodic progress log while we wait for PX4 to finish landing.
            if self.tick % max(1, int(self.rate_hz)) == 0:
                z_up = (-self.local_pos.z) if self.local_pos else float('nan')
                self.get_logger().info(
                    f'LANDING in progress: t={elapsed_s:.1f}s, '
                    f'altitude={z_up:.2f} m'
                )

            # Safety net: if PX4 never reports DISARMED, force a disarm and
            # shut down so the process always terminates eventually.
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
            # Reached terminal state without an outstanding shutdown request
            # (e.g. if a future code path forgets to call _request_shutdown).
            self._request_shutdown('STATE_DONE reached')
            return

    # ---------------------------------------------------------------------
    # State entry helpers
    # ---------------------------------------------------------------------
    def _initial_setpoint_sample(self) -> TrajectorySample:
        """Sample used during INIT/ARMING – hover at ground (the CSV origin).

        With ``do_takeoff=True`` the CSV has been shifted up so CSV[0] is at
        the takeoff altitude. We therefore manually undo that shift here to
        keep the vehicle on the ground while we wait for ARM/OFFBOARD.
        """
        sp = self._interp(0.0)
        sp.vel_ned = np.zeros(3)
        sp.acc_ned = np.zeros(3)
        sp.yawspeed_rad = 0.0
        if self.do_takeoff:
            sp.pos_ned = sp.pos_ned.copy()
            sp.pos_ned[2] += self.takeoff_altitude_m  # back to ground (NED)
        return sp

    def _takeoff_setpoint_sample(self) -> TrajectorySample:
        """Hover setpoint used during the TAKEOFF stage (CSV[0] in NED)."""
        if self._takeoff_hover_ned is None:
            self._takeoff_hover_ned = self._pos[0].copy()
            self._takeoff_yaw_rad = float(self._yaw[0])
        return TrajectorySample(
            t=0.0,
            pos_ned=self._takeoff_hover_ned.copy(),
            vel_ned=np.zeros(3),
            acc_ned=np.zeros(3),
            yaw_rad=self._takeoff_yaw_rad,
            yawspeed_rad=0.0,
        )

    def _enter_takeoff(self) -> None:
        self.takeoff_start_wall_ns = self.get_clock().now().nanoseconds
        self.off_ground_wall_ns = None
        self._takeoff_hover_ned = self._pos[0].copy()
        self._takeoff_yaw_rad = float(self._yaw[0])
        self.state = self.STATE_TAKEOFF
        self.get_logger().info(
            f'Armed + OFFBOARD; taking off to '
            f'{self.takeoff_altitude_m:.2f} m above ground (NED z = '
            f'{self._takeoff_hover_ned[2]:.2f}). Waiting for '
            f'altitude > {self.off_ground_threshold_m:.2f} m.'
        )

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
        dx = float(self.local_pos.x) - float(target_ned[0])
        dy = float(self.local_pos.y) - float(target_ned[1])
        dz = float(self.local_pos.z) - float(target_ned[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

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
        description='Play a tvc_traj_opt_gui CSV as PX4 TrajectorySetpoint messages.',
    )
    parser.add_argument(
        'csv_path',
        nargs='?',
        default=None,
        help=(
            'Path to trajectory CSV from tvc_traj_opt_gui '
            '(default: newest *.csv in TVC-traj-opt/trajs/)'
        ),
    )
    parser.add_argument('--rate', type=float, default=None, help='Publish rate [Hz]')
    parser.add_argument(
        '--time-scale', type=float, default=None,
        help='Time scaling factor (>1 slows the trajectory down).',
    )
    parser.add_argument(
        '--start-velocity-threshold', type=float, default=None,
        help='Speed [m/s] below which playback is triggered after takeoff.',
    )
    parser.add_argument(
        '--takeoff-altitude', type=float, default=None,
        help='Hover altitude above ground during the takeoff stage [m].',
    )
    parser.add_argument(
        '--no-takeoff', action='store_true',
        help='Skip the separate takeoff stage and play from CSV[0] directly.',
    )
    parser.add_argument(
        '--no-land', action='store_true',
        help='Do not send LAND after trajectory ends',
    )
    cli_args = parser.parse_args()

    csv_path = cli_args.csv_path or _default_trajectory_csv()
    if not csv_path:
        parser.error(
            'csv_path is required: pass a file path or export a CSV to TVC-traj-opt/trajs/'
        )

    overrides: dict = {'csv_path': csv_path}
    if cli_args.rate is not None:
        overrides['publish_rate_hz'] = cli_args.rate
    if cli_args.time_scale is not None:
        overrides['time_scale'] = cli_args.time_scale
    if cli_args.start_velocity_threshold is not None:
        overrides['start_velocity_threshold_m_s'] = cli_args.start_velocity_threshold
    if cli_args.takeoff_altitude is not None:
        overrides['takeoff_altitude_m'] = cli_args.takeoff_altitude
    if cli_args.no_takeoff:
        overrides['do_takeoff'] = False
    if cli_args.no_land:
        overrides['land_after_done'] = False

    sys.exit(main(param_overrides=overrides) or 0)
