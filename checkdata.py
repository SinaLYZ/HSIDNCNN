import h5py
import numpy as np
from pathlib import Path
import random

# -----------------------------
# paths
# -----------------------------
h5_path = Path("./hsi_27.h5")
train_dir = Path("HsiData/trainset")
val_dir = Path("HsiData/testset")

# -----------------------------
# settings
# -----------------------------
n_val = 12
seed = 0
save_dtype = np.float32   # better for training than float64

# -----------------------------
# create output folders
# -----------------------------
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# find all 3D cubes in the h5 file
# -----------------------------
dataset_names = []

with h5py.File(h5_path, "r") as f:
    def collect_datasets(name, obj):
        if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
            dataset_names.append(name)
            print(name, obj.shape, obj.dtype)

    f.visititems(collect_datasets)

print(f"\nFound {len(dataset_names)} total 3D cubes in {h5_path.name}")

if len(dataset_names) == 0:
    raise ValueError("No 3D datasets were found in the HDF5 file.")

# -----------------------------
# shuffle and split
# -----------------------------
random.seed(seed)
random.shuffle(dataset_names)

n_val = min(n_val, len(dataset_names))
val_names = dataset_names[:n_val]
train_names = dataset_names[n_val:]

print(f"Train cubes: {len(train_names)}")
print(f"Val cubes:   {len(val_names)}")

# -----------------------------
# save each cube as a separate .npy file
# -----------------------------
with h5py.File(h5_path, "r") as f:
    # save training cubes
    for name in train_names:
        arr = f[name][()]   # read dataset
        out_name = Path(name).name
        out_path = train_dir / out_name
        np.save(out_path, arr.astype(save_dtype))

    # save validation cubes
    for name in val_names:
        arr = f[name][()]
        out_name = Path(name).name
        out_path = val_dir / out_name
        np.save(out_path, arr.astype(save_dtype))

# -----------------------------
# final check
# -----------------------------
train_files = sorted(train_dir.glob("*.npy"))
val_files = sorted(val_dir.glob("*.npy"))

print("\nDone.")
print(f"Saved train files: {len(train_files)}")
print(f"Saved val files:   {len(val_files)}")

if len(train_files) > 0:
    x = np.load(train_files[0])
    print("\nExample training file:")
    print("Path :", train_files[0])
    print("Shape:", x.shape)
    print("Dtype:", x.dtype)
    print("Min  :", x.min())
    print("Max  :", x.max())