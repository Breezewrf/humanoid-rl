import numpy as np
import json
import os
import argparse

def remap_motion_data(input_path, output_path):
    # 1. 定义 Source (输入数据) 的关节顺序
    source_joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
    ]

    # 2. 定义 Target (目标 XML) 的关节顺序
    target_joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"
    ]

    print(f"Loading data from: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    # 加载原始数据
    data = np.load(input_path)
    # 获取数据长度 T
    T = data['joint_pos'].shape[0]
    
    # 初始化新的 joint_pos 和 joint_vel 数组
    # 形状为 [T, len(target_joint_names)]
    new_joint_pos = np.zeros((T, len(target_joint_names)))
    new_joint_vel = np.zeros((T, len(target_joint_names)))
    
    # 原始数据提取
    src_joint_pos = data['joint_pos']
    src_joint_vel = data['joint_vel']

    print("\n--- 开始重映射 (Remapping) ---")
    
    # 3. 遍历目标关节，进行数据迁移
    for i, target_name in enumerate(target_joint_names):
        if target_name in source_joint_names:
            # Case A: 关节存在于源数据中 -> 直接复制
            src_idx = source_joint_names.index(target_name)
            new_joint_pos[:, i] = src_joint_pos[:, src_idx]
            new_joint_vel[:, i] = src_joint_vel[:, src_idx]
            # print(f"  [Keep] {target_name} (Source Idx: {src_idx} -> Target Idx: {i})")
        else:
            # Case B: 关节不存在（如 wrist_roll）-> 填充 0
            # 注意：如果 XML 中默认位置不是 0，这里可能需要手动调整，但在 motion retargeting 中通常填 0 即可
            new_joint_pos[:, i] = 0.0
            new_joint_vel[:, i] = 0.0
            print(f"  [Fill Zero] {target_name} (Not in source, filling with 0.0)")

    # 4. 检查被丢弃的关节 (仅仅为了打印提示)
    dropped_joints = [j for j in source_joint_names if j not in target_joint_names]
    if dropped_joints:
        print(f"\n[Dropped] 以下关节在 Source 中存在但被过滤掉了: {dropped_joints}")

    # 5. 保存数据
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保持其他非关节数据不变 (body_pos_w, body_quat_w 等)
    # 注意：如果 Target 的 body 数量或定义与 Source 不同，body_pos_w 可能也会不准确。
    # 但通常 RL 训练主要依赖 joint_pos，或者只用 root state。
    # 这里直接透传其他所有 keys。
    save_dict = dict(data)
    save_dict['joint_pos'] = new_joint_pos
    save_dict['joint_vel'] = new_joint_vel
    
    # 强制保存
    np.savez_compressed(output_path, **save_dict)
    print(f"\nSuccess! Remapped motion saved to: {output_path}")
    print(f"New Joint Pos Shape: {new_joint_pos.shape}")

    # 6. 生成配套的 meta.json
    meta_path = os.path.join(output_dir, "meta.json")
    meta_data = {
        "joint_names": target_joint_names,
        "fps": 30.0, # 默认 30，如果你知道确切的 FPS 可以修改
        # "body_names": ... # 如果你有 target body names 也可以加进去
    }
    
    # 尝试从原目录读取 fps
    src_dir = os.path.dirname(input_path)
    src_meta = os.path.join(src_dir, "meta.json")
    if os.path.exists(src_meta):
        try:
            with open(src_meta, 'r') as f:
                old_meta = json.load(f)
                if "fps" in old_meta:
                    meta_data["fps"] = old_meta["fps"]
                if "body_names" in old_meta:
                     # 注意：如果 body 结构变了，这里的 body_names 其实也该变，
                     # 但通常为了代码不崩，先保留原来的或者根据 xml 生成。
                     # 鉴于只少了 waist 内部关节，body 列表通常主要看 Link，可能影响不大。
                     meta_data["body_names"] = old_meta["body_names"] 
        except:
            pass

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=4)
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap motion data from source to target joint configuration")
    parser.add_argument("input_folder", type=str, help="Path to input folder containing motion.npz")
    args = parser.parse_args()

    # Construct input file path
    input_folder = args.input_folder
    INPUT_FILE = os.path.join(input_folder, "motion.npz")
    
    # Create output folder with "_remapped" suffix
    folder_name = os.path.basename(input_folder.rstrip('/'))
    output_folder = os.path.join(os.path.dirname(input_folder), f"{folder_name}_remapped")
    OUTPUT_FILE = os.path.join(output_folder, "motion.npz")

    remap_motion_data(INPUT_FILE, OUTPUT_FILE)