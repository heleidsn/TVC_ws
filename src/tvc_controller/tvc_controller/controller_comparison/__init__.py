"""
控制器对比模块
用于对比 LQR 和 PID 控制器的性能
"""

from .rocket_dynamics import RocketDynamics, PhyParams
from .lqr_full_state import LQRFullStateController
from .lqr_attitude_only import LQRAttitudeOnlyController
from .pid_controller import PIDController
from .comparison_simulator import ComparisonSimulator

__all__ = [
    'RocketDynamics',
    'PhyParams',
    'LQRFullStateController',
    'LQRAttitudeOnlyController',
    'PIDController',
    'ComparisonSimulator',
]
