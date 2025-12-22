import numpy as np
import mujoco
import mujoco.viewer
import os
import time
import argparse

MODEL_PATH = "/home/breeze/Desktop/workplace/Humanoid/humanoid-rl/sim2real/data/robots/g1/g1_23dof.xml"
NPZ_FILE_PATH = "/home/breeze/Desktop/workplace/Humanoid/humanoid-rl/sim2real/data/motion/g1/Walk_B15_-_Walk_turn_around_23dof/motion.npz"

JOINT_DOF = 29  # 关节自由度数量
ROOT_BODY_INDEX = 0 # 假设第一个 body (索引 0) 是根部/骨盆

def visualize_full_motion(model_path, npz_path):
    """
    加载NPZ数据（包括根部姿态和关节位置）并在MuJoCo中可视化。
    """
    if not os.path.exists(model_path):
        print(f"❌ 错误：未找到MuJoCo模型文件：{model_path}")
        return

    # 1. 加载 MuJoCo 模型和数据
    print(f"载入模型：{model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 2. 加载 NPZ 数据
    try:
        npz_data = np.load(npz_path)
    except FileNotFoundError:
        print(f"❌ 错误：未找到NPZ文件：{npz_path}")
        return
    
    # 提取关键数组
    joint_pos = npz_data['joint_pos']      # 形状 (N, 29)
    body_pos_w = npz_data['body_pos_w']    # 形状 (N, 30, 3)
    body_quat_w = npz_data['body_quat_w']  # 形状 (N, 30, 4)
    
    # 提取 FPS，如果 NPZ 中没有，则默认为 50
    fps = npz_data['fps'][0] if 'fps' in npz_data and npz_data['fps'].size > 0 else 50
    render_delay = 1.0 / fps
    
    # 检查 MuJoCo qpos 长度是否与数据匹配
    # qpos 长度 = 7 (根部) + 29 (关节) = 36
    EXPECTED_QPOS_LENGTH = 7 + JOINT_DOF 
    if model.nq != EXPECTED_QPOS_LENGTH:
        print(f"⚠️ 警告：模型期望 qpos 长度为 {model.nq}，但数据期望长度为 {EXPECTED_QPOS_LENGTH} (7 + 29)。")
        print("请确保您的 XML 模型中 Root 的自由度为 7，且关节数恰好是 29。")
    
    print(f"帧率 (FPS): {fps}")
    print(f"总帧数: {joint_pos.shape[0]}")
    
    # 3. 使用 MuJoCo Viewer 可视化
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        
        for frame_idx in range(joint_pos.shape[0]):
            
            # --- 步骤 A: 设置根部位置和方向 (Root Pos + Quat) ---
            # qpos[0:3] = 根部位置 (x, y, z)
            # qpos[3:7] = 根部方向 (qw, qx, qy, qz)
            
            # 提取根部（例如骨盆）的世界坐标系姿态
            root_pos = body_pos_w[frame_idx, ROOT_BODY_INDEX, :]
            root_quat = body_quat_w[frame_idx, ROOT_BODY_INDEX, :]
            
            # 设置到 MuJoCo 的 qpos
            data.qpos[0:3] = root_pos 
            # 注意：MuJoCo 四元数顺序是 (w, x, y, z)，请确保您的 NPZ 数据是这个顺序
            data.qpos[3:7] = root_quat
            
            # --- 步骤 B: 设置所有关节位置 ---
            # qpos[7:] = 29 个关节位置
            current_joint_pos = joint_pos[frame_idx]
            data.qpos[7 : 7 + JOINT_DOF] = current_joint_pos
            
            # --- 可选：设置关节速度 (joint_vel) ---
            # 运行物理仿真通常不需要直接设置速度，但如果需要更准确的重放，可以使用：
            # if 'joint_vel' in npz_data:
            #     current_joint_vel = npz_data['joint_vel'][frame_idx]
            #     # 假设根部速度也需要设置，这里简单假设 root vel/ang_vel 为 0 或需要从其他数据中提取
            #     # data.qvel[6 : 6 + JOINT_DOF] = current_joint_vel # qvel 长度通常比 qpos 少一个维度 (四元数)
            
            # 执行一步 MuJoCo 仿真，将 qpos 传播到所有 body 的几何体
            mujoco.mj_forward(model, data) 
            
            viewer.sync()
            time.sleep(render_delay)

        print("运动播放完毕。")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize motion from NPZ file in MuJoCo")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to MuJoCo model XML file")
    parser.add_argument("--npz", type=str, default=NPZ_FILE_PATH, help="Path to NPZ motion file")
    args = parser.parse_args()
    JOINT_DOF = 29 if "23dof" not in args.model else 23  # 关节自由度数量
    print(f"使用的关节自由度数量: {JOINT_DOF}")
    visualize_full_motion(args.model, args.npz)