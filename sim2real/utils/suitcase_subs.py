import zmq
from scipy.spatial.transform import Rotation as R
import json
import time

def quaternion_to_rpy(x, y, z, w):
    r = R.from_quat([x, y, z, w])
    return r.as_euler('xyz', degrees=False) 

class SuitcaseSubsriber:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.ZMQ_ADDRESS = "192.168.123.164" 
        self.ZMQ_PORT = "8555"
        self.ZMQ_URL = f"tcp://{self.ZMQ_ADDRESS}:{self.ZMQ_PORT}"
        
        print(f"连接到 ZMQ Publisher，地址: {self.ZMQ_URL}...")
        self.socket.connect(self.ZMQ_URL)
        self.socket.subscribe(b"")  # 订阅所有消息

        self.socket.setsockopt(zmq.RCVTIMEO, 100) # 100 毫秒的接收超时

    def get_tf_data(self, target_frame: str, source_frame: str, timeout_sec: float = 0.5):
        """
        查询并返回指定帧之间的转换数据，格式为: 
        [None, None, [[secs, nsecs, x, y, z, roll, pitch, yaw]]]
        
        注意: 此方法通过设置 ZMQ 接收超时来模拟 TF 查询的超时。
        它返回的是**最近接收到的**匹配的 TF。
        """
        start_time = time.time()
        timeout_ms = int(timeout_sec * 1000)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms) # 设置接收超时

        tf_found = None
        
        while time.time() - start_time < timeout_sec:
            try:
                # 尝试接收消息
                message_bytes = self.socket.recv()
                message_string = message_bytes.decode('utf-8')
                tf_data = json.loads(message_string)
                
                # ZMQ 接收到的消息通常是 child frame 到 parent frame 的转换
                if tf_data['parent'] == target_frame and tf_data['child'] == source_frame:
                    # 如果数据流是 tf2 格式的 (Parent <- Child)，这通常表示 tf.lookup_transform(target_frame, source_frame)
                    # 然而在 ROS TF2 中，target_frame 通常是查询结果的 Parent，source_frame 是 Child。
                    # 我们遵循您提供的 TF 数据结构，假设：
                    # target_frame: tf_data['parent']
                    # source_frame: tf_data['child']
                    
                    translation = tf_data['translation']
                    rotation = tf_data['rotation']
                    
                    roll, pitch, yaw = quaternion_to_rpy(
                        rotation['x'], rotation['y'], rotation['z'], rotation['w']
                    )
                    
                    current_time = time.time()
                    secs = int(current_time)
                    nsecs = int((current_time - secs) * 1e9)
                    
                    tf_list = [
                        secs, 
                        nsecs, 
                        translation['x'], 
                        translation['y'], 
                        translation['z'], 
                        roll, 
                        pitch, 
                        yaw
                    ]
                    
                    tf_found = [None, None, [tf_list]]
                    return tf_found
                
                
            except zmq.error.Again:
                pass
            except zmq.error.ZMQError as e:
                print(f"ZMQ 连接错误: {e}")
                break
            except Exception as e:
                print(f"处理消息时发生错误: {e}")
                break
        
        print(f"在 {timeout_sec} 秒内未找到 TF {target_frame} -> {source_frame}。")
        return None

    def __del__(self):
        """析构函数，确保 ZMQ 资源被释放"""
        print("关闭 ZMQ Socket 和 Context。")
        self.socket.close()
        self.context.term()

# 示例使用
if __name__ == '__main__':
    # 请确保您有一个 ZMQ Publisher 正在 192.168.123.164:5555 上发送数据
    # 且数据结构与第一个代码片段中解析的一致。

    # 模拟 Publisher 发送的数据：
    # tf_data = {
    #     'parent': 'world',
    #     'child': 'camera_link',
    #     'translation': {'x': 1.0, 'y': 2.0, 'z': 3.0},
    #     'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0} # 单位四元数
    # }

    try:
        subs = SuitcaseSubsriber()

        # 尝试查询 'world' 到 'camera_link' 的转换
        # 这要求 publisher 正在发送 parent='world', child='camera_link' 的消息
        tf_result = subs.get_tf_data(
            target_frame='camera_color_optical_frame', 
            source_frame='tag_22', 
            timeout_sec=5.0
        )

        if tf_result:
            print("\n--- 成功获取转换数据 ---")
            print(f"格式: [None, None, [[secs, nsecs, x, y, z, roll, pitch, yaw]]]")
            print(tf_result)
        else:
            print("\n--- 未能获取转换数据 ---")

    except KeyboardInterrupt:
        print("\n程序停止。")
    finally:
        # 手动调用析构函数（虽然 Python 会自动管理）
        del subs