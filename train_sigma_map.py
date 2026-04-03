import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Model.DncNNSigmaMap import DnCNNSigmaMap
from Data.BandDnCNNDatasetSigmaMap import BandDnCNNDatasetSigmaMap
from utils import set_seed, save_checkpoint, get_device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    log_interval: int = 50,
) -> float:
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader, start=1):
        model_input = batch["noisy"].to(device)   # [B, 2, H, W] = [noisy_band, sigma_map]
        sigma_map = batch["sigma_map"].to(device) # [B, 1, H, W]
        true_noise = batch["noise"].to(device)    # [B, 1, H, W]

        optimizer.zero_grad()

        predicted_noise = model(model_input)      # [B, 1, H, W]
        loss = criterion(predicted_noise, true_noise)

        if batch_idx == 1 and epoch == 1:
            print(f"Input shape:     {model_input.shape}")
            print(f"Target shape:    {true_noise.shape}")
            print(f"Sigma map shape: {sigma_map.shape}")
            print(f"Predicted shape: {predicted_noise.shape}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * model_input.size(0)

        if batch_idx % log_interval == 0 or batch_idx == 1 or batch_idx == len(dataloader):
            print(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] "
                f"Batch [{batch_idx:04d}/{len(dataloader):04d}] "
                f"Loss: {loss.item():.6f}"
            )

    return running_loss / len(dataloader.dataset)


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            model_input = batch["noisy"].to(device)   # [B, 2, H, W]
            sigma_map = batch["sigma_map"].to(device)
            true_noise = batch["noise"].to(device)    # [B, 1, H, W]

            predicted_noise = model(model_input)
            loss = criterion(predicted_noise, true_noise)

            running_loss += loss.item() * model_input.size(0)

    return running_loss / len(dataloader.dataset)


def main() -> None:
    # =========================
    # Configuration
    # =========================
    seed = 42

    in_channels = 2
    out_channels = 1
    depth = 17
    num_features = 64
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-3
    num_workers = 0

    checkpoint_dir = "checkpoints"
    best_model_path = os.path.join(checkpoint_dir, "dncnn_sigma_map_best.pth")
    last_model_path = os.path.join(checkpoint_dir, "dncnn_sigma_map_last.pth")

    # =========================
    # Setup
    # =========================
    os.makedirs(checkpoint_dir, exist_ok=True)

    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    # =========================
    # Data
    # =========================
    train_dir = "processed_data/train"
    val_dir = "processed_data/val"

    train_dataset = BandDnCNNDatasetSigmaMap(
        data_dir=train_dir,
        band_sigmas=band_sigmas,
        training=True,
        patch_size=64,
    )

    val_dataset = BandDnCNNDatasetSigmaMap(
        data_dir=val_dir,
        band_sigmas=band_sigmas,
        training=False,
        patch_size=64,
        seed=seed,
    )

    print(train_dataset[0]["noisy"].shape)
    print(train_dataset[0]["noise"].shape)
    print(train_dataset[0]["clean"].shape)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Batch size:    {batch_size}")
    print(f"In channels:   {in_channels}")
    print(f"Out channels:  {out_channels}")
    print(f"Sigma mode:    band-dependent with sigma map")

    # =========================
    # Model
    # =========================
    model = DnCNNSigmaMap(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        num_features=num_features,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # =========================
    # Training
    # =========================
    best_val_loss = float("inf")
    train_history = []
    val_history = []
    lr_history = []

    print("Starting training...\n")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            log_interval=50,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        train_history.append(train_loss)
        val_history.append(val_loss)
        lr_history.append(current_lr)

        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.6e}"
        )

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            last_model_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            print(f"  -> Best model saved to: {best_model_path}")

    np.save(os.path.join(checkpoint_dir, "train_history_sigma_map.npy"), np.array(train_history))
    np.save(os.path.join(checkpoint_dir, "val_history_sigma_map.npy"), np.array(val_history))
    np.save(os.path.join(checkpoint_dir, "lr_history_sigma_map.npy"), np.array(lr_history))

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model path: {best_model_path}")


if __name__ == "__main__":
    main()