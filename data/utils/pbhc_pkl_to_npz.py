#!/usr/bin/env python3
"""
将特定PKL格式（包含 root_trans_offset, pose_aa, dof 等）转换为 HDMI 项目使用的 npz 格式

输入 PKL 格式 keys:
- root_trans_offset: 根部位置 [T, 3]
- pose_aa: 姿态轴角表示 [T, N_dims] (通常前3维是根部旋转)
- dof: 关节角度 [T, N_joints]
- root_rot: 根部旋转 [T, 4] (可选)
- smpl_joints: SMPL关节信息 (可选)
- fps: 帧率

输出 NPZ 格式 (HDMI):
- body_pos_w: 世界坐标系下的身体位置 [T, N_bodies, 3]
- body_quat_w: 世界坐标系下的身体四元数 [T, N_bodies, 4] (wxyz格式)
- joint_pos: 关节位置 [T, N_joints]
- body_lin_vel_w: 身体线速度 [T, N_bodies, 3]
- body_ang_vel_w: 身体角速度 [T, N_bodies, 3]
- joint_vel: 关节速度 [T, N_joints]

Example usage:
python3 pkl2npz.py --pkl_path /home/breeze/Desktop/workplace/Humanoid/humanoid-rl/data/motion/dance_motion_data/Charleston_dance.pkl --output_dir ./output/Charleston_dance --robot_name unitree_g1
"""

import pickle
import numpy as np
import json
import argparse
import os
from pathlib import Path
from scipy.spatial.transform import Rotation

body_names = []
joint_names = []

def compute_velocities(positions, dt):
    """计算速度（使用中心差分）"""
    velocities = np.zeros_like(positions)
    # 前向差分
    velocities[0] = (positions[1] - positions[0]) / dt
    # 中心差分
    for i in range(1, len(positions) - 1):
        velocities[i] = (positions[i + 1] - positions[i - 1]) / (2 * dt)
    # 后向差分
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    return velocities


def quaternion_to_angular_velocity(quaternions, dt):
    """从四元数序列计算角速度"""
    angular_velocities = np.zeros((len(quaternions), 3))
    for i in range(len(quaternions) - 1):
        q1 = Rotation.from_quat(quaternions[i])
        q2 = Rotation.from_quat(quaternions[i + 1])
        q_rel = q2 * q1.inv()
        rotvec = q_rel.as_rotvec()
        angular_velocities[i] = rotvec / dt
    angular_velocities[-1] = angular_velocities[-2]
    return angular_velocities


def compute_fk_mujoco(root_pos, root_rot_wxyz, dof_pos, N_bodies, T, xml_path=None):
    """
    使用 MuJoCo 计算世界坐标系下的身体位置和姿态
    返回:
        body_pos_w: [T, N_bodies, 3]
        body_quat_w: [T, N_bodies, 4] (wxyz)
    """
    import warnings
    
    # 初始化输出数组
    body_pos_w = np.zeros((T, N_bodies, 3))
    body_quat_w = np.zeros((T, N_bodies, 4))
    
    # 默认填充根部信息作为 fallback
    for t in range(T):
        for b in range(N_bodies):
            body_pos_w[t, b] = root_pos[t]
            body_quat_w[t, b] = root_rot_wxyz[t]

    if xml_path is None or not os.path.exists(xml_path):
        warnings.warn(f"Robot XML not found at {xml_path}. FK calculation will be skipped (returning root pose only).")
        return body_pos_w, body_quat_w
    try:
        import mujoco as mj
        from mujoco import viewer
        
        print(f"Loading MuJoCo model from: {xml_path}")
        model = mj.MjModel.from_xml_path(xml_path)
        data = mj.MjData(model)
        
        # Create viewer for visualization
        print("Opening MuJoCo viewer...")
        with viewer.launch_passive(model, data) as v:
            nq = model.nq

            global joint_names
            joint_names = []
            for i in range(model.njnt):
                joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
                if joint_name:
                    joint_names.append(joint_name)
            print(f"MuJoCo model joint names: {joint_names}")
            
            for t in range(T):
                qpos = np.zeros(nq, dtype=float)
                
                if nq >= 7:
                    print(f"Setting root and joint positions for frame {t}")
                    qpos[0:3] = root_pos[t]
                    print(f"  Root Pos: {qpos[0:3]}")
                    qpos[3:7] = root_rot_wxyz[t]
                    
                    n_j = min(len(dof_pos[t]), max(0, nq - 7))
                    qpos[7:7 + n_j] = dof_pos[t, :n_j]
                else:
                    n_j = min(len(dof_pos[t]), nq)
                    qpos[:n_j] = dof_pos[t, :n_j]

                # 前向运动学计算
                data.qpos[:qpos.shape[0]] = qpos
                mj.mj_forward(model, data)

                global body_names
                if t == 0:  # Only extract body names once
                    body_names = []
                    for i in range(data.xpos.shape[0]):
                        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
                        body_names.append(name)
                
                # 读取结果
                current_n_bodies = min(N_bodies, data.xpos.shape[0])
                
                # Skip the first body (world)
                body_pos_w[t, 1:current_n_bodies, :] = data.xpos[1:current_n_bodies, :]
                body_quat_w[t, 1:current_n_bodies, :] = data.xquat[1:current_n_bodies, :]
                
                # Update viewer
                v.sync()

    except ImportError:
        warnings.warn("mujoco package not installed. Cannot run FK.")
    except Exception as e:
        warnings.warn(f"FK execution failed: {e}")
    return body_pos_w, body_quat_w

def convert_pkl_to_npz(pkl_path, output_dir, robot_name="unitree_g1", base_xml_path=None):
    """
    转换主函数
    """
    print(f"加载PKL文件: {pkl_path}")
    try:
        with open(pkl_path, 'rb') as f:
            motion_data = pickle.load(f)
    except Exception as e:
        print(f"pickle读取失败，尝试使用joblib/torch: {e}")
        try:
            import joblib
            motion_data = joblib.load(pkl_path)
        except:
            import torch
            motion_data = torch.load(pkl_path, map_location="cpu")

    motion_data = motion_data[next(iter(motion_data.keys()))]
    # 1. 提取基础数据
    # ---------------------------------------------------------
    fps = motion_data.get("fps", 30.0)
    dt = 1.0 / fps
    
    # 提取根部位置
    if "root_trans_offset" in motion_data:
        root_pos = np.array(motion_data["root_trans_offset"]) # [T, 3]
    else:
        raise ValueError("Missing key 'root_trans_offset'")

    # 提取关节数据 (dof)
    if "dof" in motion_data:
        dof_pos = np.array(motion_data["dof"]) # [T, N_joints]
    else:
        raise ValueError("Missing key 'dof'")

    # 提取或计算根部旋转 (Root Rotation)
    # 优先使用 root_rot, 否则从 pose_aa 解析
    if "root_rot" in motion_data and motion_data["root_rot"] is not None:
        raw_root_rot = np.array(motion_data["root_rot"])
        # 检查形状判断是 quaternion 还是 axis-angle
        if raw_root_rot.shape[-1] == 4:
            # 假设输入是 xyzw (scipy默认) 或 wxyz? 
            # 通常 GMR/SMPL pipeline 输出 xyzw。我们需要确认。
            # 这里假设输入是 xyzw，稍后统一转为 wxyz (Mujoco格式)
            r = Rotation.from_quat(raw_root_rot)
            root_quat_xyzw = raw_root_rot
            root_rot_wxyz = r.as_quat()[:, [3, 0, 1, 2]] # xyzw -> wxyz
        elif raw_root_rot.shape[-1] == 3:
            # Axis-angle
            r = Rotation.from_rotvec(raw_root_rot)
            root_quat_xyzw = r.as_quat()
            root_rot_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    elif "pose_aa" in motion_data:
        print("未找到 'root_rot'，从 'pose_aa' 提取根部旋转...")
        pose_aa = np.array(motion_data["pose_aa"])
        root_aa = pose_aa[:, :3] # 前3维通常是根部旋转
        r = Rotation.from_rotvec(root_aa)
        root_quat_xyzw = r.as_quat()
        root_rot_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    else:
        raise ValueError("无法找到根部旋转数据 (root_rot 或 pose_aa)")

    T = root_pos.shape[0]
    N_joints = dof_pos.shape[1]
    
    # 设定身体数量 (N_bodies)
    # 注意：如果不运行 FK，这个数字只是占位。如果运行 FK，它应该匹配 XML 中的 body 数量
    # G1 robot 通常有 ~20-30 个 bodies (links)
    if robot_name == "unitree_g1":
        # 这是一个估计值，Mujoco读取时会自动截断或填充
        N_bodies = 25
    else:
        N_bodies = N_joints + 1 # 粗略估计

    print(f"数据统计:")
    print(f"  Root Pos: {root_pos.shape}")
    print(f"  DOF Pos: {dof_pos.shape}")
    print(f"  FPS: {fps}")

    # 2. 计算 FK (得到世界坐标下的 body position 和 orientation)
    # ---------------------------------------------------------
    xml_path = base_xml_path
    print("开始计算前向运动学 (FK)...")
    body_pos_w, body_quat_w = compute_fk_mujoco(
        root_pos, root_rot_wxyz, dof_pos, N_bodies, T, xml_path
    )

    # 3. 计算速度
    # ---------------------------------------------------------
    print("计算速度...")
    body_lin_vel_w = compute_velocities(body_pos_w, dt)
    joint_vel = compute_velocities(dof_pos, dt)
    
    # 计算角速度
    body_ang_vel_w = np.zeros((T, N_bodies, 3))
    for b in range(N_bodies):
        # body_quat_w 是 wxyz，计算角速度时需要注意顺序
        # quaternion_to_angular_velocity 内部使用 scipy (xyzw)，需要转换
        quat_xyzw = body_quat_w[:, b, [1, 2, 3, 0]] 
        body_ang_vel_w[:, b] = quaternion_to_angular_velocity(quat_xyzw, dt)

    # 4. 生成 Meta 信息
    # ---------------------------------------------------------
    global body_names
    global joint_names
    # 5. 保存
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "motion.npz")
    
    print(f"保存 NPZ: {npz_path}")
    np.savez_compressed(
        npz_path,
        body_pos_w=body_pos_w,       # [T, N_bodies, 3]
        body_quat_w=body_quat_w,     # [T, N_bodies, 4] (wxyz)
        joint_pos=dof_pos,           # [T, N_joints]
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_vel=joint_vel
    )
    print(f"Saved body_pos_w shape: {body_pos_w.shape}, body_quat_w shape: {body_quat_w.shape}, joint_pos shape: {dof_pos.shape}, joint_vel shape: {joint_vel.shape}, body_lin_vel_w shape: {body_lin_vel_w.shape}, body_ang_vel_w shape: {body_ang_vel_w.shape}")
    meta_path = os.path.join(output_dir, "meta.json")
    meta_data = {
        "body_names": body_names,
        "joint_names": joint_names,
        "fps": float(fps)
    }
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=4)
        
    print("转换完成。")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True, help="Input PKL path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--robot_name", type=str, default="unitree_g1")
    parser.add_argument("--base_xml_path", type=str, default="/home/breeze/Desktop/workplace/Humanoid/humanoid-rl/active_adaptation/assets_mjcf/g1_29dof_nohand/g1_23dof_lock_wrist.xml", help="Base path to robot XML files")
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"File not found: {args.pkl_path}")
        
    convert_pkl_to_npz(args.pkl_path, args.output_dir, args.robot_name, args.base_xml_path)

if __name__ == "__main__":
    main()