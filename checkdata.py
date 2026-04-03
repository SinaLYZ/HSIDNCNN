import h5py

h5_path = "Data/hsi_27.h5"

with h5py.File(h5_path, "r") as f:
    print("Top-level keys:", list(f.keys()))

    group = f["hsi_27"]
    print("Inside hsi_27:", list(group.keys()))

    for key in group.keys():
        item = group[key]
        print(f"{key}: type = {type(item)}")
        if isinstance(item, h5py.Dataset):
            print(f"    shape = {item.shape}, dtype = {item.dtype}")