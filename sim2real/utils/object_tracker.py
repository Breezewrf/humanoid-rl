import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from odom_subs import OdometrySubscriber
from suitcase_subs import SuitcaseSubsriber
from common import ZMQPublisher, PORTS

def euler_to_matrix(x, y, z, roll, pitch, yaw):
    # 使用 Scipy 的 Rotation
    # r = R.from_euler('ZYX', [roll, pitch, yaw], degrees=False)
    r = R.from_euler('ZYX', [yaw, pitch, roll], degrees=False)
    
    # 3x3 旋转矩阵
    rotation_matrix = r.as_matrix()
    
    # 4x4 齐次变换矩阵
    matrix = np.identity(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = [x, y, z]
    return matrix

cam_x, cam_y, cam_z = 0.04364, 0.0325, 0.50668
cam_roll, cam_pitch, cam_yaw = -2.40157305074, 0, -1.57079632679
  
T_odom_pelvis = np.array([[0, 0, 1, 0],
                          [-1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]])  

test = np.array([[1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]])  

class ObjectTracker:
    def __init__(self):
        self.suitcase_sub = SuitcaseSubsriber()
        self.target = "camera_color_optical_frame"
        self.source = "tag_22"
        self.T_pelvis_camera = euler_to_matrix(
            cam_x, cam_y, cam_z, cam_roll, cam_pitch, cam_yaw
        )
        self.odom_sub = OdometrySubscriber(interface="enp58s0")
        for _i in range(10):
            self.world_initial_odom = self.odom_sub.get_pose()
            time.sleep(0.1)
        
        self.world_new_initial_odom = np.identity(4)
        # self.world_new_initial_odom[:3, :3] = np.array([[0, -1, 0],
        #                                                 [0, 0, 1],
        #                                                 [-1, 0, 0]])  
        self.world_new_initial_odom[:3, 3] = [0, 0, self.world_initial_odom[2, 3]]
        
        self.last_suitcase_pose = None
        self.last_suicase_quat = None
        self.last_pelvis_pose = None
        self.last_pelvis_quat = None

        self.suitcase_publisher = ZMQPublisher(port=PORTS["suitcase_pose"])
        self.pelvis_publisher = ZMQPublisher(port=PORTS["pelvis_pose"])


    def get_position(self, object_name):
        if object_name == "suitcase":
            try:
                result = self.suitcase_sub.get_tf_data(self.target, self.source)
                if result:
                    secs, nsecs, x, y, z, roll, pitch, yaw = result[2][0]
                    # x, y, z = [i*1000 for i in [x, y, z]]
                    T_camera_suitcase = euler_to_matrix(x, y, z, roll, pitch, yaw)
                    T_world_odom = self.odom_sub.get_pose()
                    T_world_new_suitcase = self.world_new_initial_odom @ np.linalg.inv(self.world_initial_odom) @ T_world_odom @ self.T_pelvis_camera @ T_camera_suitcase

                    final_x, final_y, final_z = T_world_new_suitcase[:3, 3]
                    rotation_matrix = T_world_new_suitcase[:3, :3]
                    r = R.from_matrix(rotation_matrix)
                    final_roll, final_pitch, final_yaw = r.as_euler('XYZ', degrees=False)

                    print(f"Suitcase Orientation in XYZ order: roll: {final_roll}, pitch: {final_pitch}, yaw: {final_yaw}")

                    # a = self.world_new_initial_odom @ np.linalg.inv(self.world_initial_odom) @ T_world_odom
                    # a_rotation = R.from_matrix(a[:3, :3])
                    # a_roll, a_pitch, a_yaw = a_rotation.as_euler('XYZ', degrees=False)
                    # print(f"In XYZ order, a_roll: {a_roll}, a_pitch: {a_pitch}, a_yaw: {a_yaw}")
                    # a_pos = a[:3, 3]
                    # print(f"a_pos: {a_pos}")

                    # b = T_world_odom
                    # b_rotation = R.from_matrix(b[:3, :3])
                    # b_roll, b_pitch, b_yaw = b_rotation.as_euler('XYZ', degrees=False)
                    # print("B:", b_roll, b_pitch, b_yaw)   

                    # print("self.world_new_initial_odom", self.world_new_initial_odom)
                    # print("Initial Odom Pose:", self.world_initial_odom)
                    # print("T_world_odom: ", T_world_odom)
                    # print("T_odom_pelvis", T_odom_pelvis)
                    # print("self.T_pelvis_camera:", self.T_pelvis_camera)
                    # print("T_camera_suitcase:", T_camera_suitcase)
                    # print("T_world_new_suitcase:", T_world_new_suitcase)
                    # rospy.loginfo("--- Transform Found ---")
                    # rospy.loginfo(f"Time: {secs + nsecs / 1e9:.6f} | Pos: ({x:.4f}, {y:.4f}, {z:.4f}) | Euler: ({roll:.4f}, {pitch:.4f}, {yaw:.4f})")
                    self.last_suitcase_pose = [None, None, [[0, 0, final_x, final_y, final_z-0.31, final_roll, final_pitch, final_yaw]]]
                    self.last_suicase_quat = r.as_quat(scalar_first=True)  # [w, x, y, z]
                self.suitcase_publisher.publish_pose(position=self.last_suitcase_pose[2][0][2:5],
                                                    quaternion=self.last_suicase_quat)
                return self.last_suitcase_pose
                
            except Exception as e:
                print(f"TF查找异常: {e}")
                return None
                
        elif object_name == "pelvis":
            try:
                # pose_matrix = self.odom_sub.get_pose()
                # x = pose_matrix[0, 3]
                # y = pose_matrix[1, 3]
                # z = pose_matrix[2, 3]
                
                # rotation_matrix = pose_matrix[:3, :3]
                # r = R.from_matrix(rotation_matrix)

                T_world_odom = self.odom_sub.get_pose()
                T_world_new_odom = self.world_new_initial_odom @ np.linalg.inv(self.world_initial_odom) @ T_world_odom

                x = T_world_new_odom[0, 3]
                y = T_world_new_odom[1, 3]
                z = T_world_new_odom[2, 3]
                
                rotation_matrix = T_world_new_odom[:3, :3]
                r = R.from_matrix(rotation_matrix)
                quaternion = r.as_quat(scalar_first=True)  # [w, x, y, z]
                # b = T_world_odom
                # b_rotation = R.from_matrix(b[:3, :3])
                # b_roll, b_pitch, b_yaw = b_rotation.as_euler('XYZ', degrees=False)
                # print("B:", b_roll, b_pitch, b_yaw)  

                roll, pitch, yaw = r.as_euler('XYZ', degrees=False)

                # print(f"Pelvis Orientation in XYZ order: roll: {roll}, pitch: {pitch}, yaw: {yaw}")
                self.last_pelvis_pose = [None, None, [[0, 0, x, y, z, roll, pitch, yaw]]]
                self.last_pelvis_quat = quaternion
                self.pelvis_publisher.publish_pose(position=self.last_pelvis_pose[2][0][2:5],
                                                  quaternion=self.last_pelvis_quat)
                return self.last_pelvis_pose
                
            except Exception as e:
                print(f"获取里程计数据异常: {e}")
                return None
        
        return None
        
    def process_position(self, vicon_object_name):
        """处理位置数据的辅助方法"""
        position = self.get_position(vicon_object_name)
        if not position:
            print(f"Cannot get the pose of {vicon_object_name}.")
            return None, None, None
        
        try:
            obj = position[2][0]
            _, _, x, y, z, roll, pitch, yaw = obj
            current_time = time.time()
            
            # 位置和方向
            position_vec = np.array([x, y, z])  # 转换为米
            # rotation = R.from_euler('ZYX', [roll, pitch, yaw], degrees=False)
            rotation = R.from_euler('ZYX', [yaw, pitch, roll], degrees=False)
            quaternion = rotation.as_quat()
            # print("roll, pitch, yaw:", roll, pitch, yaw)
            quaternion = np.roll(quaternion, 1)  # [x, y, z, w] -> [w, x, y, z]
            
            return current_time, position_vec, quaternion
            
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None, None, None

if __name__ == "__main__":
    tracker = ObjectTracker()
    print("Starting object tracking...")
    while True:
        time_stamp, position, quaternion = tracker.process_position("suitcase")
        if position is not None:
            print(f"Tracking suitcase, Time: {time_stamp}, Position: {position}, Quaternion: {quaternion}")
        time_stamp, position, quaternion = tracker.process_position("pelvis")
        if position is not None:
            print(f"Tracking pelvis, Time: {time_stamp}, Position: {position}, Quaternion: {quaternion}")
        # time.sleep(1.0)