import numpy as np
import json
import os

def filter_data(npz_path, json_path, output_path):
    # Load the data
    data = np.load(npz_path)
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Get the lists from metadata
    body_names = metadata['body_names']
    joint_names = metadata['joint_names']
    
    # Define the indices to remove
    remove_link_list = [
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    ]
    remove_joint_list = [
        "waist_roll_joint",
        "waist_pitch_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]

    body_indices_to_remove = []
    for link_name in remove_link_list:
        try:
            body_indices_to_remove.append(body_names.index(link_name))
        except ValueError:
            print(f"Warning: {link_name} not found in body_names.")

    joint_indices_to_remove = []
    for joint_name in remove_joint_list:
        try:
            joint_indices_to_remove.append(joint_names.index(joint_name))
        except ValueError:
            print(f"Warning: {joint_name} not found in joint_names.")
    
    # Create new filtered arrays
    filtered_data = {}
    
    # Filter body-related arrays (removing specific body indices)
    body_arrays = ['body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w']
    for array_name in body_arrays:
        if array_name in data.files:
            # Keep all frames, remove specified body indices, keep all dimensions
            mask = np.ones(data[array_name].shape[1], dtype=bool)
            mask[body_indices_to_remove] = False
            filtered_data[array_name] = data[array_name][:, mask, :]
    
    # Filter joint-related arrays (removing specific joint indices)
    joint_arrays = ['joint_pos', 'joint_vel']
    for array_name in joint_arrays:
        if array_name in data.files:
            # Keep all frames, remove specified joint indices
            mask = np.ones(data[array_name].shape[1], dtype=bool)
            mask[joint_indices_to_remove] = False
            filtered_data[array_name] = data[array_name][:, mask]
    
    # Keep object_contact array as is
    if 'object_contact' in data.files:
        filtered_data['object_contact'] = data['object_contact']
    
    # Filter metadata
    filtered_metadata = metadata.copy()
    filtered_metadata['body_names'] = [name for i, name in enumerate(metadata['body_names']) 
                                     if i not in body_indices_to_remove]
    filtered_metadata['joint_names'] = [name for i, name in enumerate(metadata['joint_names']) 
                                      if i not in joint_indices_to_remove]
    
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
    base_path = os.path.expanduser("/home/breeze/Desktop/workplace/Humanoid/HDMI/data/motion/g1/walk1_subject1-3_12")
    output_path = os.path.expanduser("/home/breeze/Desktop/workplace/Humanoid/HDMI/data/motion/g1/walk1_subject1-3_12_23dof")
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
        
    filter_data(
        npz_path=os.path.join(base_path, "motion.npz"),
        json_path=os.path.join(base_path, "meta.json"),
        output_path=output_path
    )
