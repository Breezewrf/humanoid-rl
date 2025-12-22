"""
里程计数据订阅器模块
"""
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# Unitree SDK 导入
UNITREE_SDK_AVAILABLE = False
try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    pass

class OdometrySubscriber:
    """里程计数据订阅器"""
    
    def __init__(self, interface: str = "enp58s0"):
        self.position = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.rpy = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.lock = threading.Lock()
        self.transform_matrix = np.eye(4)
        self.velocity = np.zeros(3)
        self.last_update_time = time.time()
        
        if not UNITREE_SDK_AVAILABLE:
            print("[WARNING] Unitree SDK 不可用，使用默认位姿")
            return
        
        try:
            ChannelFactoryInitialize(0, interface)
            self.odom_subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)
            self.odom_subscriber.Init(self._odom_handler, 10)
            print(f"[INFO] 里程计订阅器已初始化，接口: {interface}")
        except Exception as e:
            print(f"[ERROR] 初始化里程计订阅器失败: {e}")
    
    def _odom_handler(self, msg: SportModeState_):
        """里程计数据处理回调"""
        try:
            with self.lock:
                current_time = time.time()
                dt = current_time - self.last_update_time
                
                # 更新位置和速度
                new_position = np.array(msg.position[:3])
                if dt > 0:
                    self.velocity = (new_position - self.position) / dt
                
                self.position = new_position
                self.last_update_time = current_time
                
                if hasattr(msg, 'imu_state') and hasattr(msg.imu_state, 'quaternion'):
                    quat = msg.imu_state.quaternion
                    self.rpy = np.array([msg.imu_state.rpy[0], msg.imu_state.rpy[1], msg.imu_state.rpy[2]])
                    self.orientation = np.array([quat[3], quat[0], quat[1], quat[2]])
                
                self.transform_matrix = self._compute_transform_matrix()
        
        except Exception as e:
            print(f"[ERROR] 里程计数据处理失败: {e}")
    
    def _compute_transform_matrix(self) -> np.ndarray:
        """计算机器人变换矩阵"""
        w, x, y, z = self.orientation
        
        # 四元数到旋转矩阵
        # R = np.array([
        #     [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        #     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        #     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        # ])
        # R = np.array([
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, 1]
        # ])
        
        r = R.from_euler('ZYX', [self.rpy[2], self.rpy[1], self.rpy[0]], degrees=False)
    
        # 3x3 旋转矩阵
        rotation_matrix = r.as_matrix()
        
        # 4x4 齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = self.position


        # T = np.eye(4)
        # T[:3, :3] = R
        # T[:3, 3] = self.position
        return T
    
    def get_pose(self) -> np.ndarray:
        """获取当前位姿变换矩阵"""
        with self.lock:
            return self.transform_matrix.copy()
    
    def get_velocity(self) -> np.ndarray:
        """获取当前速度"""
        with self.lock:
            return self.velocity.copy()
        

if __name__ == "__main__":
    odom_sub = OdometrySubscriber(interface="enp58s0")
    while True:
        pose = odom_sub.get_pose()
        vel = odom_sub.get_velocity()
        print(f"Pose:\n{pose}\nVelocity: {vel}\n")
        time.sleep(1.0)