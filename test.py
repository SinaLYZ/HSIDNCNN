import os

import torch
from torch.utils.data import DataLoader

from Model.DncNN import DnCNN
from Data.BandDnCNNDataset import BandDnCNNDataset
from utils import calculate_psnr, save_numpy_image, get_device, load_checkpoint


def main() -> None:
    # =========================
    # Configuration
    # =========================
    test_dir = "processed_data/test"
    checkpoint_path = "checkpoints/dncnn_best.pth"
    results_dir = "results"

    in_channels = 1
    depth = 17
    num_features = 64
    sigma = 25
    batch_size = 1
    num_workers = 0
    save_outputs = True

    # =========================
    # Setup
    # =========================
    device = get_device()
    os.makedirs(results_dir, exist_ok=True)

    print(f"Using device: {device}")

    # =========================
    # Data
    # =========================
    test_dataset = BandDnCNNDataset(
        data_dir=test_dir,
        sigma=sigma,
        training=False,
        patch_size=64,
        seed=42,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Checkpoint:   {checkpoint_path}")

    # =========================
    # Model
    # =========================
    model = DnCNN(
        in_channels=in_channels,
        depth=depth,
        num_features=num_features,
    ).to(device)

    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
    )
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")

    # =========================
    # Testing
    # =========================
    total_psnr_noisy = 0.0
    total_psnr_denoised = 0.0
    num_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            key = batch["key"][0] if isinstance(batch["key"], list) else batch["key"]

            predicted_noise = model(noisy)
            denoised = noisy - predicted_noise
            denoised = torch.clamp(denoised, 0.0, 1.0)

            psnr_noisy = calculate_psnr(noisy, clean, max_value=1.0)
            psnr_denoised = calculate_psnr(denoised, clean, max_value=1.0)

            total_psnr_noisy += psnr_noisy
            total_psnr_denoised += psnr_denoised
            num_samples += 1

            print(
                f"[{i + 1:03d}/{len(test_loader):03d}] "
                f"Key: {key} | "
                f"PSNR Noisy: {psnr_noisy:.2f} dB | "
                f"PSNR Denoised: {psnr_denoised:.2f} dB"
            )

            if save_outputs:
                sample_dir = os.path.join(results_dir, str(key))
                os.makedirs(sample_dir, exist_ok=True)

                save_numpy_image(noisy[0], os.path.join(sample_dir, "noisy.npy"))
                save_numpy_image(clean[0], os.path.join(sample_dir, "clean.npy"))
                save_numpy_image(predicted_noise[0], os.path.join(sample_dir, "predicted_noise.npy"))
                save_numpy_image(denoised[0], os.path.join(sample_dir, "denoised.npy"))

    avg_psnr_noisy = total_psnr_noisy / num_samples
    avg_psnr_denoised = total_psnr_denoised / num_samples

    print("\nTesting finished.")
    print(f"Average PSNR of noisy images:    {avg_psnr_noisy:.2f} dB")
    print(f"Average PSNR of denoised images: {avg_psnr_denoised:.2f} dB")
    print(f"Average PSNR improvement:        {avg_psnr_denoised - avg_psnr_noisy:.2f} dB")


if __name__ == "__main__":
    main()