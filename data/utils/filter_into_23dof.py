import numpy as np
import json
import os
import argparse

def filter_data(npz_path, json_path, output_path):
    # Load the data
    data = np.load(npz_path)
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Get the lists from metadata
    body_names = metadata['body_names']
    joint_names = metadata['joint_names']
    
    # Define the bodies and joints to keep (23 DOF)
    bodies_to_keep = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",  # Maps to "left_wrist_roll_rubber_hand"
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link"   # Maps to "right_wrist_roll_rubber_hand"
    ]
    
    joints_to_keep = [
        "root",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint"
    ]
    
    # Get indices of bodies and joints to keep
    body_indices_to_keep = [
        i for i, name in enumerate(body_names) 
        if name in bodies_to_keep or 
           (name == "left_wrist_roll_rubber_hand" and "left_wrist_roll_link" in bodies_to_keep) or
           (name == "right_wrist_roll_rubber_hand" and "right_wrist_roll_link" in bodies_to_keep)
    ]
    
    joint_indices_to_keep = [
        i for i, name in enumerate(joint_names) 
        if name in joints_to_keep
    ]
    
    # Create new filtered arrays
    filtered_data = {}
    
    # Filter body-related arrays (keeping only specific body indices)
    body_arrays = ['body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w']
    for array_name in body_arrays:
        if array_name in data.files:
            # Keep all frames, keep only specified body indices, keep all dimensions
            filtered_data[array_name] = data[array_name][:, body_indices_to_keep, :]
    
    # Filter joint-related arrays (keeping only specific joint indices)
    joint_arrays = ['joint_pos', 'joint_vel']
    for array_name in joint_arrays:
        if array_name in data.files:
            # Keep all frames, keep only specified joint indices
            filtered_data[array_name] = data[array_name][:, joint_indices_to_keep]
    
    # Keep object_contact array as is
    if 'object_contact' in data.files:
        filtered_data['object_contact'] = data['object_contact']
    
    # Filter metadata
    filtered_metadata = metadata.copy()
    filtered_metadata['body_names'] = [name for i, name in enumerate(metadata['body_names']) 
                                     if i in body_indices_to_keep]
    filtered_metadata['joint_names'] = [name for i, name in enumerate(metadata['joint_names']) 
                                      if i in joint_indices_to_keep]
    
    # Save filtered data
    np.savez(output_path + '/motion.npz', **filtered_data)
    with open(output_path + '/meta.json', 'w') as f:
        json.dump(filtered_metadata, f, indent=2)
    
    # Print information about the filtered data
    print("\nFiltered data shapes:")
    for key, array in filtered_data.items():
        print(f"{key}: {array.shape}")
    
    print("\nNumber of DOFs after filtering:")
    print(f"Body DOFs: {len(filtered_metadata['body_names'])}")
    print(f"Joint DOFs: {len(filtered_metadata['joint_names'])}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter motion data into 23 DOF format.")
    parser.add_argument(
        "--base-path", "-b",
        default="/home/breeze/Desktop/workplace/Humanoid/HDMI/data/utils/output/tennis/",
        help="Path to folder containing motion.npz and meta.json"
    )
    parser.add_argument(
        "--output-path", "-o",
        default="/home/breeze/Desktop/workplace/Humanoid/HDMI/data/utils/output/tennis_converted_23dof",
        help="Directory to write filtered motion.npz and meta.json"
    )
    args = parser.parse_args()

    base_path = os.path.expanduser(args.base_path)
    output_path = os.path.expanduser(args.output_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filter_data(
        npz_path=os.path.join(base_path, "motion.npz"),
        json_path=os.path.join(base_path, "meta.json"),
        output_path=output_path
    )