#!/usr/bin/env python3
"""
Convert GMR PKL motion format → NPZ format for HDMI project
FK calculated directly from MuJoCo using the Unitree G1 XML model.

GMR input format (per frame):
    - root_pos: [T, 3]
    - root_rot: [T, 4] (xyzw)
    - dof_pos:  [T, N_joints]
    - fps: float

Output NPZ:
    - body_pos_w      [T, N_bodies, 3]
    - body_quat_w     [T, N_bodies, 4] (wxyz)
    - joint_pos       [T, N_joints]
    - body_lin_vel_w  [T, N_bodies, 3]
    - body_ang_vel_w  [T, N_bodies, 3]
    - joint_vel       [T, N_joints]

Meta:
    - body_names
    - joint_names
    - fps
"""

import pickle
import numpy as np
import json
import argparse
import os
from pathlib import Path
import mujoco as mj


# ---------------------------------------------------------
# Util: Fix quaternion continuity (avoid ± flips)
# ---------------------------------------------------------
def fix_quaternion_continuity(quats):
    """
    Ensure quaternion sequence has consistent signs to avoid jumps.
    Input: quats [T,4] (wxyz)
    """
    fixed = quats.copy()
    for i in range(1, len(quats)):
        if np.dot(fixed[i], fixed[i-1]) < 0:
            fixed[i] = -fixed[i]
    return fixed


# ---------------------------------------------------------
# Velocity utilities (central difference)
# ---------------------------------------------------------
def compute_velocities(data, dt):
    vel = np.zeros_like(data)
    vel[0] = (data[1] - data[0]) / dt
    vel[-1] = (data[-1] - data[-2]) / dt
    for i in range(1, len(data)-1):
        vel[i] = (data[i+1] - data[i-1]) / (2*dt)
    return vel


# ---------------------------------------------------------
# Main converter
# ---------------------------------------------------------
def convert_pkl_to_npz(pkl_path, output_dir):
    print(f"Loading PKL: {pkl_path}")
    with open(pkl_path, "rb") as f:
        motion_data = pickle.load(f)

    fps = float(motion_data["fps"])
    dt = 1.0 / fps

    root_pos = motion_data["root_pos"]       # [T,3]
    root_rot_xyzw = motion_data["root_rot"]  # [T,4] xyzw
    dof_pos = motion_data["dof_pos"]         # [T,N]

    T = root_pos.shape[0]
    N_joints = dof_pos.shape[1]

    print("Frames:", T)
    print("Joints:", N_joints)
    print("FPS:", fps)

    # ------------------------------
    # Load Unitree G1 MuJoCo model
    # ------------------------------
    xml_path = '/home/breeze/Desktop/workplace/Humanoid/GMR/assets/unitree_g1/g1_mocap_29dof.xml'

    if not os.path.exists(xml_path):
        raise FileNotFoundError("G1 XML not found: " + xml_path)

    print(f"Loading MuJoCo XML: {xml_path}")
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    N_bodies = model.nbody
    print("Bodies:", N_bodies)

    # Allocate output arrays
    body_pos_w = np.zeros((T, N_bodies, 3))
    body_quat_w = np.zeros((T, N_bodies, 4))  # wxyz
    body_lin_vel_w = np.zeros((T, N_bodies, 3))
    body_ang_vel_w = np.zeros((T, N_bodies, 3))

    # ------------------------------
    # Prepare input quaternions
    # MuJoCo expects wxyz
    # ------------------------------
    root_quat_wxyz = np.zeros_like(root_rot_xyzw)
    root_quat_wxyz[:, 0] = root_rot_xyzw[:, 3]
    root_quat_wxyz[:, 1:] = root_rot_xyzw[:, :3]
    root_quat_wxyz = fix_quaternion_continuity(root_quat_wxyz)

    print("Evaluating FK...")

    nq = model.nq
    for t in range(T):

        # Construct qpos = [root_pos(3), root_quat(4), joint_pos...]
        qpos = np.zeros(nq)

        qpos[0:3] = root_pos[t]
        qpos[3:7] = root_quat_wxyz[t]

        n_j = min(N_joints, nq - 7)
        qpos[7:7+n_j] = dof_pos[t, :n_j]

        data.qpos[:] = qpos
        mj.mj_forward(model, data)

        # Copy FK results
        body_pos_w[t] = data.xpos.copy()              # [nbody,3]
        body_quat_w[t] = data.xquat.copy()            # [nbody,4], already wxyz
        body_lin_vel_w[t] = data.cvel[:, 0:3].copy()  # linear vel
        body_ang_vel_w[t] = data.cvel[:, 3:6].copy()  # angular vel

    # Joint velocity
    joint_vel = compute_velocities(dof_pos, dt)

    # ------------------------------
    # Save output
    # ------------------------------
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "motion.npz")

    print("Saving:", npz_path)
    np.savez_compressed(
        npz_path,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_pos=dof_pos,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_vel=joint_vel
    )

    # Names from MuJoCo model
    body_names = [model.body(i).name for i in range(N_bodies)]
    joint_names = [model.joint(i).name for i in range(model.njnt)]

    meta_path = os.path.join(output_dir, "meta.json")
    print("Saving meta:", meta_path)

    meta = {
        "body_names": body_names,
        "joint_names": joint_names,
        "fps": fps
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print("Done!")


# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    convert_pkl_to_npz(args.pkl_path, args.output_dir)


if __name__ == "__main__":
    main()
