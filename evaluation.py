import os
import numpy as np
import matplotlib.pyplot as plt


def psnr(img1, img2, max_value=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_value ** 2) / mse)


# =========================
# Part 1: Image evaluation
# =========================
sample_dir = "results/20230627114143_band_000"   # change this if needed

clean = np.load(os.path.join(sample_dir, "clean.npy"))
noisy = np.load(os.path.join(sample_dir, "noisy.npy"))
denoised = np.load(os.path.join(sample_dir, "denoised.npy"))

clean = np.squeeze(clean)
noisy = np.squeeze(noisy)
denoised = np.squeeze(denoised)

noisy_error = np.abs(clean - noisy)
denoised_error = np.abs(clean - denoised)

psnr_noisy = psnr(noisy, clean)
psnr_denoised = psnr(denoised, clean)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].imshow(clean, cmap="gray")
axes[0, 0].set_title("Clean")
axes[0, 0].axis("off")

axes[0, 1].imshow(noisy, cmap="gray")
axes[0, 1].set_title(f"Noisy\nPSNR = {psnr_noisy:.2f} dB")
axes[0, 1].axis("off")

axes[0, 2].imshow(denoised, cmap="gray")
axes[0, 2].set_title(f"Denoised\nPSNR = {psnr_denoised:.2f} dB")
axes[0, 2].axis("off")

im1 = axes[1, 0].imshow(noisy_error, cmap="hot")
axes[1, 0].set_title("Absolute Error: Noisy")
axes[1, 0].axis("off")
fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

im2 = axes[1, 1].imshow(denoised_error, cmap="hot")
axes[1, 1].set_title("Absolute Error: Denoised")
axes[1, 1].axis("off")
fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

axes[1, 2].bar(["Noisy", "Denoised"], [psnr_noisy, psnr_denoised])
axes[1, 2].set_title("PSNR Comparison")
axes[1, 2].set_ylabel("PSNR (dB)")

plt.tight_layout()
plt.show()


# =========================
# Part 2: Training curves
# =========================
train_history = np.load("checkpoints/train_history.npy")
val_history = np.load("checkpoints/val_history.npy")
lr_history = np.load("checkpoints/lr_history.npy")

epochs = np.arange(1, len(train_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_history, label="Train Loss")
plt.plot(epochs, val_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(epochs, lr_history)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid(True)
plt.tight_layout()
plt.show()