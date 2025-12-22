import numpy as np

def summary(npz_path):
    d = np.load(npz_path)
    print("File:", npz_path)
    for k in d.files:
        a = d[k]
        print(k, "shape", a.shape, "dtype", a.dtype)
        if np.issubdtype(a.dtype, np.number):
            print("  min", np.nanmin(a), "max", np.nanmax(a), "mean", np.nanmean(a), "std", np.nanstd(a))
            # detect all-zero
            if np.allclose(a, 0):
                print("  >> all zeros!")
        # for quaternions check norms
        if k == 'body_quat_w':
            # assume shape (T, N, 4)
            norms = np.linalg.norm(a.reshape(-1,4), axis=1)
            print("  quat norms min/max/mean:", norms.min(), norms.max(), norms.mean())
            # check if same quaternion repeated across joints for first frame
            if a.shape[0] > 0:
                first = a[0]  # (N,4)
                unique_rows = np.unique(first.round(6), axis=0)
                print("  unique quats in first frame:", unique_rows.shape[0], "of", first.shape[0])
        # for joint_vel, print percentiles
        if k == 'joint_vel':
            flat = a.reshape(-1)
            pct = np.percentile(np.abs(flat), [50, 90, 95, 99, 100])
            print("  abs joint_vel percentiles (50,90,95,99,100):", pct)
            # list top outliers
            out = np.where(np.abs(flat) > 10)[0]  # threshold 10 rad/s suspicious
            print("  Count abs(vel)>10:", out.size)
    d.close()

# Example: run for each file path
summary("/home/breeze/Desktop/workplace/Humanoid/humanoid-rl/data/motion/g1/Walk_B15_-_Walk_turn_around_23dof/motion.npz")
summary("/home/breeze/Desktop/workplace/Humanoid/humanoid-rl/data/motion/g1/walk1_subject1-3_12_23dof/motion.npz")
summary("/home/breeze/Desktop/workplace/Humanoid/humanoid-rl/data/motion/g1/omomo/sub1_suitcase_011_23dof/motion.npz")
