import h5py

h5_path = "Data/hsi_27.h5"


def print_h5_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"[Group]   {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"[Dataset] {name} | shape={obj.shape} | dtype={obj.dtype}")

with h5py.File(h5_path, "r") as f:
    f.visititems(print_h5_structure)