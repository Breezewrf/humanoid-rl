import numpy as np
import json
import os

# Define the path to your data
base_path = os.path.expanduser("/home/breeze/Desktop/workplace/Humanoid/HDMI/data/motion/g1/walk1_subject1-3_12")
npz_file = os.path.join(base_path, "motion.npz")
json_file = os.path.join(base_path, "meta.json")

# Read the NPZ file
def read_npz_data(file_path):
    try:
        data = np.load(file_path)
        print("\nNPZ file contents:")
        print("Available arrays:", data.files)
        
        # Print information about each array in the NPZ file
        for key in data.files:
            array = data[key]
            print(f"\nArray '{key}':")
            print(f"Shape: {array.shape}")
            print(f"Data type: {array.dtype}")
            print(f"First few elements: {array.flatten()[:50]}")
            
        return data
    except Exception as e:
        print(f"Error reading NPZ file: {e}")
        return None

# Read the JSON metadata
def read_metadata(file_path):
    try:
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        print("\nMetadata contents:")
        print(json.dumps(metadata, indent=2))
        return metadata
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

if __name__ == "__main__":
    # Read both files
    npz_data = read_npz_data(npz_file)
    metadata = read_metadata(json_file)
