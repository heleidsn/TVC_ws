#!/usr/bin/env python3
"""
Full-State LQR Controller

This module implements a full-state LQR controller that controls all 12 states:
position, velocity, attitude, and angular velocity.
Similar to the lqr_controller_node implementation.
"""

import numpy as np
import scipy.linalg
from typing import Optional
import sys
import os

# Handle both relative and absolute imports
if __name__ == "__main__" or __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from controller_comparison.rocket_dynamics import PhyParams, RocketDynamics
else:
    from .rocket_dynamics import PhyParams, RocketDynamics


class LQRFullStateController:
    """
    Full-state LQR controller for TVC rocket.
    
    State vector (12D): [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
    Control input (4D): [phi, theta, thrust, tau_r]
    
    Coordinate System:
    - Input states: ENU (East-North-Up) - from RocketDynamics
    - Internal LQR computation: NED (North-East-Down) - aligned with original lqr_controller_node
    - Output control: Same for both (phi, theta, thrust, tau_r don't need conversion)
    """
    
    def __init__(self, phy_params: PhyParams, Q: Optional[np.ndarray] = None, 
                 R: Optional[np.ndarray] = None, use_ned: bool = True):
        """
        Initialize full-state LQR controller.
        
        Args:
            phy_params: Physical parameters
            Q: State weighting matrix (12x12), if None uses default
            R: Control weighting matrix (4x4), if None uses default
            use_ned: If True, use NED coordinate system for LQR computation (aligned with original)
                    If False, use ENU coordinate system (current RocketDynamics)
        """
        self.params = phy_params
        self.dynamics = RocketDynamics(phy_params)
        self.use_ned = use_ned
        
        if use_ned:
            # Use NED coordinate system (aligned with original lqr_controller_node)
            # Linearize in NED frame
            self.A, self.B = self._linearize_ned()
        else:
            # Use ENU coordinate system (current RocketDynamics)
            # Linearize around equilibrium
            self.A, self.B = self.dynamics.linearize()
        
        # Set default Q and R if not provided
        if Q is None:
            Q_diag = np.array([1.0, 1.0, 1.0,  # position
                               1.0, 1.0, 1.0,  # velocity
                               1.0, 1.0, 0.1,  # attitude (lower weight on qz)
                               1.0, 1.0, 0.01])  # angular velocity (lower weight on r)
            self.Q = np.diag(Q_diag)
        else:
            self.Q = Q
            
        if R is None:
            R_diag = np.array([10.0, 10.0, 1.0, 10.0])  # [phi, theta, thrust, tau_r]
            self.R = np.diag(R_diag)
        else:
            self.R = R
        
        # Solve LQR
        self.K = None
        self.P = None
        self.solve_lqr()
    
    def solve_lqr(self) -> bool:
        """
        Solve the continuous-time LQR problem.
        
        Solves: A'P + PA - PBR^(-1)B'P + Q = 0
        Gain: K = R^(-1)B'P
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            self.K = np.linalg.solve(self.R, self.B.T @ self.P)
            return True
        except Exception as e:
            print(f"Error solving LQR: {e}")
            self.K = None
            self.P = None
            return False
    
    def _enu_to_ned(self, state_enu: np.ndarray) -> np.ndarray:
        """
        Convert state from ENU (East-North-Up) to NED (North-East-Down).
        
        Args:
            state_enu: State vector in ENU frame [x_enu, y_enu, z_enu, vx_enu, vy_enu, vz_enu, qx, qy, qz, p, q, r]
            
        Returns:
            State vector in NED frame [x_ned, y_ned, z_ned, vx_ned, vy_ned, vz_ned, qx, qy, qz, p, q, r]
        """
        state_ned = state_enu.copy()
        
        # Position: ENU -> NED
        # NED: x_ned = y_enu (North), y_ned = x_enu (East), z_ned = -z_enu (Down = -Up)
        state_ned[0] = state_enu[1]   # x_ned = y_enu (North)
        state_ned[1] = state_enu[0]   # y_ned = x_enu (East)
        state_ned[2] = -state_enu[2]  # z_ned = -z_enu (Down = -Up)
        
        # Velocity: ENU -> NED
        # NED: vx_ned = vy_enu (North velocity), vy_ned = vx_enu (East velocity), vz_ned = -vz_enu (Down = -Up)
        state_ned[3] = state_enu[4]   # vx_ned = vy_enu (North velocity)
        state_ned[4] = state_enu[3]   # vy_ned = vx_enu (East velocity)
        state_ned[5] = -state_enu[5]  # vz_ned = -vz_enu (Down = -Up)
        
        # Attitude and angular velocity: unchanged (qx, qy, qz, p, q, r)
        # state_ned[6:12] = state_enu[6:12]  # Already copied
        
        return state_ned
    
    def _linearize_ned(self) -> tuple:
        """
        Linearize the dynamics in NED coordinate system (aligned with original lqr_controller_node).
        
        Returns:
            A: 12x12 state matrix in NED frame
            B: 12x4 control input matrix in NED frame
        """
        # NED linearization (based on original lqr.py implementation)
        # State vector: [x_ned, y_ned, z_ned, vx_ned, vy_ned, vz_ned, qx, qy, qz, p, q, r]
        # Control input: [phi, theta, thrust, tau_r]
        
        g = self.params.G
        mass = self.params.MASS
        I_XX = self.params.I_XX
        I_YY = self.params.I_YY
        I_ZZ = self.params.I_ZZ
        l = self.params.DIST_COM_2_THRUST
        
        # Linearized A matrix (12x12) in NED frame
        A = np.zeros((12, 12))
        
        # Position dynamics: pos_dot = vel
        A[0:3, 3:6] = np.eye(3)
        
        # Velocity dynamics: vel_dot = f(attitude, thrust)
        # For small angles: qx ≈ roll, qy ≈ pitch
        # In NED: vx_dot (North) ≈ -2g * qy (pitch), vy_dot (East) ≈ 2g * qx (roll)
        A[3, 7] = -2 * g  # vx_dot (North) ≈ -2g * qy (pitch)
        A[4, 6] = 2 * g   # vy_dot (East) ≈ 2g * qx (roll)
        # vz_dot (Down): handled in B matrix (thrust control)
        
        # Quaternion kinematics: q_vec_dot ≈ 0.5 * omega (for small angles)
        A[6, 9] = 0.5   # qx_dot = 0.5 * p
        A[7, 10] = 0.5  # qy_dot = 0.5 * q
        A[8, 11] = 0.5  # qz_dot = 0.5 * r
        
        # Angular velocity dynamics: omega_dot = f(torque)
        # Handled in B matrix
        
        # Linearized B matrix (12x4) in NED frame
        B = np.zeros((12, 4))
        
        # Velocity dynamics: affected by thrust deflection and magnitude
        # Control input definition (consistent with ENU):
        #   phi (u[0]) affects East direction
        #   theta (u[1]) affects North direction
        # In NED: vx_ned = vy_enu (North), vy_ned = vx_enu (East)
        # So: phi affects vy_ned (East, index 4), theta affects vx_ned (North, index 3)
        # Sign analysis:
        #   - In ENU: phi > 0 → vx_dot < 0 (East deceleration, B[3,0] = -g)
        #   - In NED: phi > 0 → vy_dot < 0 (East deceleration, since vy_ned = vx_enu)
        #   - Therefore: B[4, 0] = -g
        #   - In ENU: theta > 0 → vy_dot > 0 (North acceleration, B[4,1] = g)
        #   - In NED: theta > 0 → vx_dot > 0 (North acceleration, since vx_ned = vy_enu)
        #   - Therefore: B[3, 1] = g
        B[3, 1] = -g       # vx_dot (North) from theta - consistent with ENU where theta affects vy (North)
        B[4, 0] = g       # vy_dot (East) from phi - consistent with ENU where phi affects vx (East)
        B[5, 2] = 1.0 / mass  # vz_dot (Down) from thrust
        
        # Angular velocity dynamics: affected by deflection and yaw torque
        B[9, 0] = -l * mass * g / I_XX   # p_dot from phi
        B[10, 1] = -l * mass * g / I_YY  # q_dot from theta
        B[11, 3] = 1.0 / I_ZZ             # r_dot from tau_r
        
        return A, B
    
    def compute_control(self, state: np.ndarray, state_ref: np.ndarray) -> np.ndarray:
        """
        Compute control input using LQR feedback law.
        
        Args:
            state: Current state in ENU frame [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
            state_ref: Reference state in ENU frame [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
            
        Returns:
            Control input [phi, theta, thrust, tau_r] (same for both ENU and NED)
        """
        if self.K is None:
            raise ValueError("LQR gain matrix K is not computed. Call solve_lqr() first.")

        self.use_ned = False
        
        if self.use_ned:
            # Convert ENU states to NED for LQR computation
            # state_ned = self._enu_to_ned(state)
            state_ned = state.copy()
            # state_ref_ned = self._enu_to_ned(state_ref)
            state_ref_ned = state_ref.copy()
            
            # Compute state error in NED frame
            error_ned = state_ned - state_ref_ned
            
            # LQR control law in NED: u = u_eq - K * error
            # However, for Z-axis in NED, the direction is opposite to ENU
            # So we need to flip the sign for Z and vz related gains
            # Create a modified error vector with Z and vz signs flipped
            error_ned_modified = error_ned.copy()

            # limit error
            # error_ned_modified[0:3] = np.clip(error_ned_modified[0:3], -0.1, 0.1)  # position error m
            # error_ned_modified[3:6] = np.clip(error_ned_modified[3:6], -0.1, 0.1)  # velocity error m/s
            # error_ned_modified[6:9] = np.clip(error_ned_modified[6:9], -0.1, 0.1)  # attitude error rad
            # error_ned_modified[9:12] = np.clip(error_ned_modified[9:12], -0.1, 0.1)  # angular velocity error rad/s

            # error_ned_modified[2] = -error_ned[2]  # Z position: flip sign
            # error_ned_modified[5] = -error_ned[5]  # Z velocity: flip sign
            
            # Apply LQR control law
            u = -self.K @ error_ned_modified

            print(f"u: {u}")
            # Add equilibrium control input
            # In NED: equilibrium thrust = mg (to balance gravity)
            # Original implementation: thrust_gimbal_z_frame = u_lqr[2] - MASS * G
            # This means u_lqr[2] = thrust_gimbal_z_frame + mg
            # So we add mg here to match the original behavior
            # u_eq = np.array([0.0, 0.0, self.params.MASS * self.params.G, 0.0])
            u = u + u_eq
        else:
            # Use ENU coordinate system (current RocketDynamics)
            # Compute state error
            error = state - state_ref
            
            # LQR control law: u = u_eq - K * error
            u = -self.K @ error
            
            # Add equilibrium control input
            u_eq = np.array([0.0, 0.0, self.params.MASS * self.params.G, 0.0])
            u = u + u_eq
        
        return u
    
    def get_gain_matrix(self) -> Optional[np.ndarray]:
        """Get the LQR gain matrix K (4x12)."""
        return self.K
    
    def get_riccati_solution(self) -> Optional[np.ndarray]:
        """Get the Riccati equation solution P (12x12)."""
        return self.P


def simulate_step_linearized(state: np.ndarray, u: np.ndarray, A: np.ndarray, B: np.ndarray, 
                             state_eq: np.ndarray, u_eq: np.ndarray, dt: float) -> np.ndarray:
    """
    使用线性化动力学进行一步积分。
    
    Args:
        state: 当前状态 [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]
        u: 控制输入 [phi, theta, thrust, tau_r]
        A: 线性化状态矩阵 (12x12)
        B: 线性化控制矩阵 (12x4)
        state_eq: 平衡点状态
        u_eq: 平衡点控制输入
        dt: 时间步长
        
    Returns:
        下一步状态
    """
    # 计算相对于平衡点的偏差
    x_dev = state - state_eq
    u_dev = u - u_eq
    
    # 线性化动力学: x_dot = A @ x_dev + B @ u_dev
    x_dot = A @ x_dev + B @ u_dev
    
    # 欧拉积分
    state_next = state + dt * x_dot
    
    # 归一化四元数（防止数值误差）
    q_vec = state_next[6:9]
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 1.0:
        state_next[6:9] = q_vec / q_norm
    elif q_norm < 0.01:  # 如果四元数太小，重置为零
        state_next[6:9] = 0.0
    
    return state_next


def main(save_plots: bool = False, use_linearized_dynamics: bool = True):
    """
    测试主函数：初始化LQR控制器，输出K矩阵，并进行仿真。
    
    Args:
        save_plots: If True, save plots to files. Default is False.
        use_linearized_dynamics: If True, use linearized dynamics (A @ x + B @ u).
                                 If False, use full nonlinear dynamics. Default is False.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    import numpy as np
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    
    # 物理参数（从默认参数）
    phy_params = PhyParams(
        MASS=0.6570,           # kg
        G=9.81,                # m/s²
        I_XX=0.062796,         # kg·m²
        I_YY=0.062976,         # kg·m²
        I_ZZ=0.001403,         # kg·m²
        DIST_COM_2_THRUST=0.5693,  # m
    )
    
    # 创建LQR控制器
    print("=" * 80)
    print("初始化LQR全状态控制器...")
    print("=" * 80)
    controller = LQRFullStateController(phy_params, use_ned=True)
    
    # 输出K矩阵
    K = controller.get_gain_matrix()
    if K is not None:
        print("\nLQR增益矩阵 K (4x12):")
        print("=" * 80)
        print("K矩阵形状:", K.shape)
        print("\nK矩阵内容:")
        print(K)
        print("\nK矩阵详细说明:")
        print("  行1 (phi控制):   对应状态 [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]")
        print("  行2 (theta控制): 对应状态 [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]")
        print("  行3 (thrust控制):对应状态 [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]")
        print("  行4 (tau_r控制): 对应状态 [x, y, z, vx, vy, vz, qx, qy, qz, p, q, r]")
        print("=" * 80)
    else:
        print("错误: 无法计算K矩阵!")
        return
    
    
    # 输出P矩阵（可选）
    P = controller.get_riccati_solution()
    if P is not None:
        print("\nRiccati方程解 P (12x12):")
        print("P矩阵形状:", P.shape)
        print("P矩阵特征值:", np.linalg.eigvals(P))
        print("=" * 80)
    
    # 设置目标位置（ENU坐标系）
    print("\n设置目标位置...")
    target_position = np.array([3.0, 0.0, 0.0])  # [x, y, z] 单位：米
    print(f"目标位置: x={target_position[0]:.2f}m, y={target_position[1]:.2f}m, z={target_position[2]:.2f}m")
    
    # 构建参考状态（目标位置，零速度和姿态）
    state_ref = np.array([
        target_position[0], target_position[1], target_position[2],  # 位置
        0.0, 0.0, 0.0,  # 速度
        0.0, 0.0, 0.0,  # 姿态（四元数向量部分）
        0.0, 0.0, 0.0   # 角速度
    ])
    
    # 初始状态（从原点开始）
    state0 = np.array([
        0.0, 0.0, 0.0,  # 位置（原点）
        0.0, 0.0, 0.0,  # 速度
        0.0, 0.0, 0.0,  # 姿态
        0.0, 0.0, 0.0   # 角速度
    ])
    
    print(f"初始位置: x={state0[0]:.2f}m, y={state0[1]:.2f}m, z={state0[2]:.2f}m")
    
    # 仿真参数
    dt = 0.01  # 时间步长 (s)
    t_end = 10.0  # 仿真时间 (s)
    n_steps = int(t_end / dt) + 1
    
    print(f"\n开始仿真...")
    print(f"时间步长: {dt}s")
    print(f"仿真时长: {t_end}s")
    print(f"仿真步数: {n_steps}")
    print("=" * 80)
    
    # 初始化动力学模型
    dynamics = RocketDynamics(phy_params)
    
    # 如果使用线性化动力学，需要获取A和B矩阵以及平衡点
    if use_linearized_dynamics:
        print("\n使用线性化动力学进行仿真...")
        # 获取线性化矩阵（使用ENU坐标系）
        A, B = dynamics.linearize()
        # 平衡点状态（悬停，无旋转）
        state_eq = np.zeros(12)
        state_eq[2] = 0.0  # z = 0 (地面高度)
        # 平衡点控制输入（推力平衡重力，无偏转）
        u_eq = np.array([0.0, 0.0, phy_params.MASS * phy_params.G, 0.0])
        print("线性化动力学矩阵已准备")
    else:
        print("\n使用完整非线性动力学进行仿真...")
        A, B = None, None
        state_eq, u_eq = None, None
    
    # 存储轨迹
    time = np.linspace(0, t_end, n_steps)
    state_traj = np.zeros((n_steps, 12))
    control_traj = np.zeros((n_steps, 4))
    state_traj[0] = state0
    
    # 控制约束
    phi_min, phi_max = -1.0, 1.0  # rad
    theta_min, theta_max = -1.0, 1.0  # rad
    thrust_min, thrust_max = 0.0, 2.0 * phy_params.MASS * phy_params.G  # N
    tau_r_min, tau_r_max = -0.5, 0.5  # Nm
    
    # 仿真循环
    state = state0.copy()
    for i in range(1, n_steps):
        # 计算控制输入
        print("----------------step {i}----------------")
        print(f"state: {state}")
        u = controller.compute_control(state, state_ref)

        print(f"u_raw: {u}")
   
        # 应用控制约束
        u[0] = np.clip(u[0], phi_min, phi_max)
        u[1] = np.clip(u[1], theta_min, theta_max)
        u[2] = np.clip(u[2], thrust_min, thrust_max)
        u[3] = np.clip(u[3], tau_r_min, tau_r_max)

        print(f"u_limited: {u}")
        
        control_traj[i] = u
        
        # 积分动力学
        if use_linearized_dynamics:
            # 使用线性化动力学: x_dot = A @ (x - x_eq) + B @ (u - u_eq)
            state = simulate_step_linearized(state, u, A, B, state_eq, u_eq, dt)
        else:
            # 使用完整非线性动力学（RK4积分）
            state = dynamics.simulate_step(state, u, dt)
        state_traj[i] = state
    
    print("仿真完成!")
    print("=" * 80)
    
    # 计算最终误差
    final_error = state_traj[-1] - state_ref
    pos_error = np.linalg.norm(final_error[0:3])
    print(f"\n最终位置误差: {pos_error:.4f}m")
    print(f"最终位置: x={state_traj[-1, 0]:.4f}m, y={state_traj[-1, 1]:.4f}m, z={state_traj[-1, 2]:.4f}m")
    print(f"目标位置: x={state_ref[0]:.4f}m, y={state_ref[1]:.4f}m, z={state_ref[2]:.4f}m")
    
    # 绘制结果
    print("\n生成图表...")
    
    # 将四元数转换为欧拉角（用于显示）
    from scipy.spatial.transform import Rotation
    def quaternion_to_euler(q_vec):
        """将四元数向量部分转换为欧拉角 [roll, pitch, yaw]"""
        qw = np.sqrt(np.clip(1.0 - np.sum(q_vec**2, axis=-1), 0.0, 1.0))
        if q_vec.ndim == 1:
            q = np.array([qw, q_vec[0], q_vec[1], q_vec[2]])
            rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            euler = rot.as_euler('ZYX', degrees=False)
            return np.array([euler[2], euler[1], euler[0]])  # [roll, pitch, yaw]
        else:
            quat_scipy = np.column_stack([q_vec, qw])
            rot = Rotation.from_quat(quat_scipy)
            euler = rot.as_euler('ZYX', degrees=False)
            return np.column_stack([euler[:, 2], euler[:, 1], euler[:, 0]])  # [roll, pitch, yaw]
    
    # 计算欧拉角轨迹
    euler_traj = quaternion_to_euler(state_traj[:, 6:9])
    euler_ref = quaternion_to_euler(state_ref[6:9]) if np.any(state_ref[6:9] != 0) else np.array([0.0, 0.0, 0.0])
    
    # 创建综合图：5行，每行3-4个子图
    fig, axes = plt.subplots(5, 4, figsize=(16, 14))
    fig.suptitle('LQR Controller Simulation Results', fontsize=16, fontweight='bold')
    
    # 第一行：位置 (x, y, z)
    pos_labels = ['X (m) - East', 'Y (m) - North', 'Z (m) - Up']
    for i in range(3):
        axes[0, i].plot(time, state_traj[:, i], 'b-', linewidth=2, label='Actual')
        axes[0, i].axhline(y=state_ref[i], color='r', linestyle='--', linewidth=2, label='Target')
        axes[0, i].set_ylabel(pos_labels[i], fontsize=11)
        axes[0, i].legend(loc='best', fontsize=9)
        axes[0, i].grid(True, alpha=0.3)
    axes[0, 3].axis('off')  # 第4列空白
    
    # 第二行：速度 (vx, vy, vz)
    vel_labels = ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
    for i in range(3):
        axes[1, i].plot(time, state_traj[:, 3+i], 'b-', linewidth=2, label='Actual')
        axes[1, i].axhline(y=state_ref[3+i], color='r', linestyle='--', linewidth=2, label='Target')
        axes[1, i].set_ylabel(vel_labels[i], fontsize=11)
        axes[1, i].legend(loc='best', fontsize=9)
        axes[1, i].grid(True, alpha=0.3)
    axes[1, 3].axis('off')  # 第4列空白
    
    # 第三行：角度 (roll, pitch, yaw)
    att_labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    for i in range(3):
        axes[2, i].plot(time, np.degrees(euler_traj[:, i]), 'b-', linewidth=2, label='Actual')
        axes[2, i].axhline(y=np.degrees(euler_ref[i]), color='r', linestyle='--', linewidth=2, label='Target')
        axes[2, i].set_ylabel(att_labels[i], fontsize=11)
        axes[2, i].legend(loc='best', fontsize=9)
        axes[2, i].grid(True, alpha=0.3)
    axes[2, 3].axis('off')  # 第4列空白
    
    # 第四行：角速度 (p, q, r)
    omega_labels = ['P (deg/s)', 'Q (deg/s)', 'R (deg/s)']
    for i in range(3):
        axes[3, i].plot(time, np.degrees(state_traj[:, 9+i]), 'b-', linewidth=2, label='Actual')
        axes[3, i].axhline(y=np.degrees(state_ref[9+i]), color='r', linestyle='--', linewidth=2, label='Target')
        axes[3, i].set_ylabel(omega_labels[i], fontsize=11)
        axes[3, i].legend(loc='best', fontsize=9)
        axes[3, i].grid(True, alpha=0.3)
    axes[3, 3].axis('off')  # 第4列空白
    
    # 第五行：控制量 (phi, theta, thrust, tau_r)
    control_labels = ['Phi (deg)', 'Theta (deg)', 'Thrust (N)', 'Tau_r (Nm)']
    for i in range(4):
        if i < 2:  # phi, theta - 转换为度
            axes[4, i].plot(time, np.degrees(control_traj[:, i]), 'g-', linewidth=2)
        else:  # thrust, tau_r
            axes[4, i].plot(time, control_traj[:, i], 'g-', linewidth=2)
        axes[4, i].set_ylabel(control_labels[i], fontsize=11)
        axes[4, i].set_xlabel('Time (s)', fontsize=11)
        axes[4, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('lqr_test_results.png', dpi=300, bbox_inches='tight')
        print("已保存: lqr_test_results.png")
    print("=" * 80)
    
    # 显示图表
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LQR Full State Controller Test')
    parser.add_argument('--save-plots', action='store_true', 
                        help='Save plots to files (default: False)')
    parser.add_argument('--linearized-dynamics', action='store_true',
                        help='Use linearized dynamics (A @ x + B @ u) instead of full nonlinear dynamics (default: False)')
    args = parser.parse_args()
    main(save_plots=args.save_plots, use_linearized_dynamics=args.linearized_dynamics)
