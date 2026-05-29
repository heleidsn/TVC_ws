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
    INIT      -> send heartbeat + setpoint at CSV[0], wait for OFFBOARD + ARMED
    ARMING    -> request OFFBOARD then ARM repeatedly until accepted
    PLAYING   -> publish setpoints by interpolating CSV at the current physical
                 time (t0 = wall-clock at entry)
    HOLDING   -> CSV done: keep publishing the last sample for a short hold
    LANDING   -> request LAND mode, then DISARM, then exit

The publisher rate is configurable (``--rate``, default 50 Hz). The heartbeat
must be sent at >=2 Hz for PX4 to stay in OFFBOARD mode; we send it every
control tick.
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
    STATE_PLAYING = 'PLAYING'
    STATE_HOLDING = 'HOLDING'
    STATE_LANDING = 'LANDING'
    STATE_DONE = 'DONE'

    def __init__(self) -> None:
        super().__init__('tvc_traj_player')

        # ---- Parameters ---------------------------------------------------
        self.declare_parameter('csv_path', '')
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('arm_wait_setpoints', 20)
        self.declare_parameter('hold_after_done_s', 2.0)
        self.declare_parameter('land_after_done', True)
        self.declare_parameter('use_acceleration_setpoint', True)
        # Optional initial-position offset (NED, meters). Useful if the
        # autopilot's home is not exactly at the CSV origin.
        self.declare_parameter('origin_offset_ned', [0.0, 0.0, 0.0])

        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        self.rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.arm_wait_setpoints = int(self.get_parameter('arm_wait_setpoints').value)
        self.hold_after_done_s = float(self.get_parameter('hold_after_done_s').value)
        self.land_after_done = bool(self.get_parameter('land_after_done').value)
        self.use_acceleration_setpoint = bool(
            self.get_parameter('use_acceleration_setpoint').value
        )
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
        self._yaw = np.deg2rad(rows_ned[:, idx['yaw_deg']]) if 'yaw_deg' in idx else np.zeros(len(self._t))
        self._acc = self._finite_diff(self._vel, self._t)
        self._yaw_rate = self._finite_diff_1d(self._yaw, self._t, wrap_angle=True)
        self.traj_duration = float(self._t[-1])

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

        # ---- State -------------------------------------------------------
        self.state = self.STATE_INIT
        self.tick = 0
        self.vehicle_status: Optional[VehicleStatus] = None
        self.local_pos: Optional[VehicleLocalPosition] = None
        self.play_start_wall_ns: Optional[int] = None
        self.hold_start_wall_ns: Optional[int] = None

        period = 1.0 / max(self.rate_hz, 1e-3)
        self.timer = self.create_timer(period, self._on_tick)

        self.get_logger().info(
            f'Player armed. Duration={self.traj_duration:.2f}s, '
            f'rate={self.rate_hz} Hz, samples={len(self._t)}, '
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
    # Tick state machine
    # ---------------------------------------------------------------------
    def _on_tick(self) -> None:
        self.tick += 1
        self._publish_heartbeat()  # required every cycle while in OFFBOARD

        if self.state == self.STATE_INIT:
            # Hold the first sample so PX4 has a valid setpoint to latch onto.
            first = self._interp(0.0)
            first.acc_ned = np.zeros(3)
            self._publish_setpoint(first)

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
            first = self._interp(0.0)
            first.acc_ned = np.zeros(3)
            self._publish_setpoint(first)
            if self.tick % int(self.rate_hz) == 0:
                # Re-issue the commands until PX4 acknowledges
                self._engage_offboard()
                self._arm()

            if self._is_armed_and_offboard():
                self.play_start_wall_ns = self.get_clock().now().nanoseconds
                self.state = self.STATE_PLAYING
                self.get_logger().info(
                    f'Armed and OFFBOARD; starting playback ({self.traj_duration:.2f}s)'
                )
            return

        if self.state == self.STATE_PLAYING:
            t_now = (self.get_clock().now().nanoseconds - self.play_start_wall_ns) / 1e9
            if t_now >= self.traj_duration:
                t_now = self.traj_duration
                self.hold_start_wall_ns = self.get_clock().now().nanoseconds
                self.state = self.STATE_HOLDING
                self.get_logger().info(
                    f'Playback complete; holding final pose for {self.hold_after_done_s:.1f}s'
                )
            self._publish_setpoint(self._interp(t_now))
            return

        if self.state == self.STATE_HOLDING:
            last = self._interp(self.traj_duration)
            last.vel_ned = np.zeros(3)
            last.acc_ned = np.zeros(3)
            last.yawspeed_rad = 0.0
            self._publish_setpoint(last)

            held_s = (self.get_clock().now().nanoseconds - self.hold_start_wall_ns) / 1e9
            if held_s >= self.hold_after_done_s:
                if self.land_after_done:
                    self._land()
                    self.state = self.STATE_LANDING
                    self.get_logger().info('Hold complete; requesting LAND.')
                else:
                    self.state = self.STATE_DONE
                    self.get_logger().info('Hold complete; stopping (land_after_done=false).')
            return

        if self.state == self.STATE_LANDING:
            if self.vehicle_status and (
                self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
            ):
                self.state = self.STATE_DONE
                self.get_logger().info('Vehicle disarmed; shutting down node.')
                rclpy.shutdown()
            return

        # STATE_DONE: do nothing

    def _is_armed_and_offboard(self) -> bool:
        if self.vehicle_status is None:
            return False
        return (
            self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
            and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = TVCTrajectoryPlayer()
    except SystemExit as e:
        rclpy.shutdown()
        sys.exit(e.code)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted; shutting down.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
