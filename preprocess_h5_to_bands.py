import os
from pathlib import Path

import h5py
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_band_minmax(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)
    band_min = band.min()
    band_max = band.max()

    if band_max > band_min:
        band = (band - band_min) / (band_max - band_min)
    else:
        band = np.zeros_like(band, dtype=np.float32)

    return band


def save_cube_bands(
    h5_path: str,
    group_name: str,
    dataset_names: list[str],
    output_dir: str,
    normalize: bool = True,
) -> None:
    ensure_dir(output_dir)

    with h5py.File(h5_path, "r") as h5_file:
        group = h5_file[group_name]

        for dataset_name in dataset_names:
            dataset_path = f"{group_name}/{dataset_name}"
            cube = np.array(group[dataset_name], dtype=np.float32)

            if cube.ndim != 3:
                raise ValueError(
                    f"Expected cube shape (H, W, B), got {cube.shape} for {dataset_path}"
                )

            h, w, num_bands = cube.shape
            print(f"Processing {dataset_name} | shape=({h}, {w}, {num_bands})")

            base_name = Path(dataset_name).stem

            for band_idx in range(num_bands):
                band = cube[:, :, band_idx]

                if normalize:
                    band = normalize_band_minmax(band)

                save_name = f"{base_name}_band_{band_idx:03d}.npy"
                save_path = os.path.join(output_dir, save_name)
                np.save(save_path, band.astype(np.float32))

            print(f"Saved {num_bands} bands to {output_dir}")


def main() -> None:
    h5_path = "Data/hsi_27.h5"
    group_name = "hsi_27"
    output_root = "processed_data"

    train_dir = os.path.join(output_root, "train")
    val_dir = os.path.join(output_root, "val")
    test_dir = os.path.join(output_root, "test")

    ensure_dir(output_root)
    ensure_dir(train_dir)
    ensure_dir(val_dir)
    ensure_dir(test_dir)

    with h5py.File(h5_path, "r") as h5_file:
        if group_name not in h5_file:
            raise KeyError(f"Group '{group_name}' not found in {h5_path}")

        all_dataset_names = sorted(list(h5_file[group_name].keys()))

    print(f"Total cubes found: {len(all_dataset_names)}")

    # ----------- SPLIT BY WHOLE CUBES -----------
    # Example split:
    # first 5 cubes -> train
    # next 1 cube   -> val
    # next 2 cubes  -> test
    train_names = all_dataset_names[:5]
    val_names = all_dataset_names[5:6]
    test_names = all_dataset_names[6:8]

    print("\nSplit summary:")
    print(f"Train cubes ({len(train_names)}): {train_names}")
    print(f"Val cubes   ({len(val_names)}): {val_names}")
    print(f"Test cubes  ({len(test_names)}): {test_names}")

    print("\nSaving training bands...")
    save_cube_bands(
        h5_path=h5_path,
        group_name=group_name,
        dataset_names=train_names,
        output_dir=train_dir,
        normalize=True,
    )

    print("\nSaving validation bands...")
    save_cube_bands(
        h5_path=h5_path,
        group_name=group_name,
        dataset_names=val_names,
        output_dir=val_dir,
        normalize=True,
    )

    print("\nSaving test bands...")
    save_cube_bands(
        h5_path=h5_path,
        group_name=group_name,
        dataset_names=test_names,
        output_dir=test_dir,
        normalize=True,
    )

    print("\nPreprocessing finished.")


if __name__ == "__main__":
    main()