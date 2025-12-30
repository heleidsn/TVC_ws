#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
from scipy.spatial.transform import Rotation as R


class OdometryNEDConverter(Node):
    """
    ROS节点：读取/model/tvc_0/odometry话题，将线速度从ENU坐标系转换到NED坐标系
    """
    
    def __init__(self):
        super().__init__('odometry_ned_converter')
        
        # 存储当前姿态四元数 [qx, qy, qz, qw]
        self.current_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        
        # 创建订阅者，订阅原始odometry数据
        self.subscription = self.create_subscription(
            Odometry,
            '/model/tvc_0/odometry',
            self.odometry_callback,
            10
        )
        
        # 创建发布者，发布转换后的NED坐标系数据
        self.ned_publisher = self.create_publisher(
            Odometry,
            '/model/tvc_0/odometry_ned',
            10
        )
        
        # 创建发布者，发布转换后的线速度
        self.velocity_publisher = self.create_publisher(
            Twist,
            '/model/tvc_0/velocity_ned',
            10
        )
        
        self.get_logger().info('Odometry NED Converter node started')
        self.get_logger().info('Subscribing to: /model/tvc_0/odometry')
        self.get_logger().info('Publishing to: /model/tvc_0/odometry_ned and /model/tvc_0/velocity_ned')
    
    def enu_to_ned_velocity(self, enu_velocity):
        """
        将ENU坐标系的线速度转换为NED坐标系
        ENU: East-North-Up
        NED: North-East-Down
        
        转换矩阵:
        N = E
        E = N  
        D = -U
        """
        ned_velocity = Twist()
        ned_velocity.linear.x = enu_velocity.linear.y  # N = E
        ned_velocity.linear.y = enu_velocity.linear.x  # E = N
        ned_velocity.linear.z = -enu_velocity.linear.z  # D = -U
        
        # 角速度保持不变
        ned_velocity.angular.x = enu_velocity.angular.x
        ned_velocity.angular.y = enu_velocity.angular.y
        ned_velocity.angular.z = enu_velocity.angular.z
        
        return ned_velocity
    
    def enu_to_ned_pose(self, enu_pose):
        """
        将ENU坐标系的位姿转换为NED坐标系
        """
        ned_pose = enu_pose
        # 位置转换
        ned_pose.position.x = enu_pose.position.y  # N = E
        ned_pose.position.y = enu_pose.position.x  # E = N
        ned_pose.position.z = -enu_pose.position.z  # D = -U
        
        # 四元数转换（简化处理，实际应用中可能需要更复杂的四元数转换）
        # 这里保持四元数不变，因为主要是处理线速度
        ned_pose.orientation = enu_pose.orientation
        
        return ned_pose
    
    def body_to_ned_velocity(self, body_velocity, current_quaternion):
        """
        将body坐标系的线速度转换为NED坐标系
        使用当前姿态四元数进行旋转转换
        """
        # 创建body坐标系下的速度向量
        body_vel_vector = np.array([
            body_velocity.linear.x,
            body_velocity.linear.y, 
            body_velocity.linear.z
        ])
        
        # 使用当前姿态四元数创建旋转对象
        # 注意：四元数格式为 [qx, qy, qz, qw]
        
        # q_gazebo = current_quaternion
        # r_gazebo = R.from_quat(q_gazebo)
        # r_flip = R.from_euler('x', 180, degrees=True)
        # r_px4 = r_flip*r_gazebo*r_flip
        # q_px4 = r_px4.as_quat()
        
        rotation = R.from_quat(current_quaternion)
        
        # 将body坐标系的速度转换到NED坐标系
        ned_vel_vector = rotation.apply(body_vel_vector)
        
        # 创建NED坐标系的速度消息
        ned_velocity = Twist()
        ned_velocity.linear.x = ned_vel_vector[1]  # North
        ned_velocity.linear.y = ned_vel_vector[0]  # East  
        ned_velocity.linear.z = -ned_vel_vector[2]  # Down
        
        # 角速度转换（如果需要的话）
        body_ang_vel_vector = np.array([
            body_velocity.angular.x,
            body_velocity.angular.y,
            body_velocity.angular.z
        ])
        
        ned_ang_vel_vector = rotation.apply(body_ang_vel_vector)
        ned_velocity.angular.x = ned_ang_vel_vector[0]
        ned_velocity.angular.y = ned_ang_vel_vector[1]
        ned_velocity.angular.z = ned_ang_vel_vector[2]
        
        return ned_velocity
    
    def odometry_callback(self, msg):
        """
        处理接收到的odometry消息
        """
        try:
            # 更新当前姿态四元数
            # 从消息中提取四元数 [qx, qy, qz, qw]
            self.current_quaternion = np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])
            
            # 创建新的NED坐标系odometry消息
            ned_odometry = Odometry()
            
            # 复制header信息
            ned_odometry.header = msg.header
            ned_odometry.header.frame_id = 'map_ned'  # 更新坐标系标识
            ned_odometry.child_frame_id = 'base_link_ned'
            
            # 转换位姿
            ned_pose = self.enu_to_ned_pose(msg.pose.pose)
            ned_odometry.pose.pose = ned_pose
            
            # 转换线速度
            ned_velocity = self.body_to_ned_velocity(msg.twist.twist, self.current_quaternion)
            ned_odometry.twist.twist = ned_velocity
            
            # 发布转换后的odometry消息
            self.ned_publisher.publish(ned_odometry)
            
            # 发布转换后的线速度
            # self.velocity_publisher.publish(ned_velocity)
            
            # 打印调试信息（可选）
            self.get_logger().debug(
                f'ENU velocity: [{msg.twist.twist.linear.x:.3f}, {msg.twist.twist.linear.y:.3f}, {msg.twist.twist.linear.z:.3f}] -> '
                f'NED velocity: [{ned_velocity.linear.x:.3f}, {ned_velocity.linear.y:.3f}, {ned_velocity.linear.z:.3f}]'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error processing odometry message: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    node = OdometryNEDConverter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
