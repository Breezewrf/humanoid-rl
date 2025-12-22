import argparse
import numpy as np
import json
import os
import torch

def read_npz_data(file_path, show_first=50, list_only=False):
    try:
        data = np.load(file_path)
        print(f"\nNPZ file: {file_path}")
        print("Available arrays:", data.files)
        if list_only:
            return data
        for key in data.files:
            array = data[key]
            print(f"\nArray '{key}': shape={array.shape}, dtype={array.dtype}")
            flat = array.flatten()
            print(f"First {min(show_first, flat.size)} elements: {flat[:show_first]}")
        return data
    except Exception as e:
        print(f"Error reading NPZ file '{file_path}': {e}")
        return None

def read_metadata(file_path):
    try:
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        print(f"\nMetadata file: {file_path}")
        print(json.dumps(metadata, indent=2))
        return metadata
    except Exception as e:
        print(f"Error reading JSON file '{file_path}': {e}")
        return None

def read_pkldata(file_path):
    try:
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"\nPKL file: {file_path}")
        if hasattr(data, "keys"):
            print("Keys:", list(data.keys()))
            for key in data.keys():
                value = data[key]
                if isinstance(value, np.ndarray):
                    print(f"\nKey '{key}': shape={value.shape}, dtype={value.dtype}")
                    print(f"First 50 elements: {value[:50]}")
                elif isinstance(value, torch.Tensor):
                    print(f"\nKey '{key}': Tensor shape={value.shape}, dtype={value.dtype}")
                    print(f"First 50 elements: {value.flatten()[:50]}")
                else:
                    print(f"\nKey '{key}': data: {value}, length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
        else:
            print("Type:", type(data))
        return data
    except Exception as e:
        print(f"Error reading PKL file '{file_path}': {e}")
        return None

def resolve_path(base_path, fn):
    if fn is None:
        return None
    if os.path.isabs(fn):
        return os.path.expanduser(fn)
    return os.path.join(base_path, fn)

if __name__ == "__main__":
    default_base = "/home/breeze/Desktop/workplace/Humanoid/GMR"
    parser = argparse.ArgumentParser(description="Read motion dataset files (npz/json/pkl).")
    parser.add_argument("--base-path", "-b", default=default_base,
                        help="Base directory for dataset (expanded).")
    parser.add_argument("--npz", "-n", default="motion.npz",
                        help="NPZ filename or absolute path (default: motion.npz in base-path).")
    parser.add_argument("--json", "-j", default=None,
                        help="JSON metadata filename or absolute path (optional).")
    parser.add_argument("--pkl", "-p", default=None,
                        help="PKL filename or absolute path (optional).")
    parser.add_argument("--show-first", "-s", type=int, default=50,
                        help="Number of elements to show from each array (default: 50).")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list arrays in the NPZ without dumping data.")
    args = parser.parse_args()

    base_path = os.path.expanduser(args.base_path)
    npz_file = resolve_path(base_path, args.npz)
    json_file = resolve_path(base_path, args.json)
    pkl_file = resolve_path(base_path, args.pkl)

    if not os.path.isdir(base_path):
        print(f"Warning: base path does not exist or is not a directory: {base_path}")

    if npz_file and os.path.exists(npz_file):
        read_npz_data(npz_file, show_first=args.show_first, list_only=args.list_only)
    else:
        print(f"NPZ file not found: {npz_file}")

    if json_file:
        if os.path.exists(json_file):
            read_metadata(json_file)
        else:
            print(f"JSON file not found: {json_file}")

    if pkl_file:
        if os.path.exists(pkl_file):
            read_pkldata(pkl_file)
        else:
            print(f"PKL file not found: {pkl_file}")
