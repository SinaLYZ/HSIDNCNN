import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Model.DncNNSigmaMap import DnCNNSigmaMap
from Data.H5DnCNNDatasetBandSigmaMap import H5DnCNNDatasetBandSigmaMap
from utils import set_seed, save_checkpoint, get_device


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    num_epochs,
    log_interval=50,
):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader, start=1):
        # Already [B, 2, H, W] = [noisy_band, sigma_map]
        model_input = batch["noisy"].to(device)
        true_noise = batch["noise"].to(device)

        optimizer.zero_grad()
        predicted_noise = model(model_input)
        loss = criterion(predicted_noise, true_noise)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * model_input.size(0)

        if batch_idx == 1 and epoch == 1:
            print(f"Input shape:        {model_input.shape}")
            print(f"Target shape:       {true_noise.shape}")
            print(f"Predicted shape:    {predicted_noise.shape}")
            print(f"Band indices:       {batch['band_idx'][:8]}")
            print(f"Raw sigma values:   {batch['sigma_raw'][:8].view(-1)}")
            print(f"Norm sigma values:  {batch['sigma'][:8].view(-1)}")

        if batch_idx % log_interval == 0 or batch_idx == 1 or batch_idx == len(dataloader):
            print(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] "
                f"Batch [{batch_idx:04d}/{len(dataloader):04d}] "
                f"Loss: {loss.item():.6f}"
            )

    return running_loss / len(dataloader.dataset)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            model_input = batch["noisy"].to(device)
            true_noise = batch["noise"].to(device)

            predicted_noise = model(model_input)
            loss = criterion(predicted_noise, true_noise)

            running_loss += loss.item() * model_input.size(0)

    return running_loss / len(dataloader.dataset)


def main():
    # -------------------------
    # Config
    # -------------------------
    seed = 42
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-3
    num_workers = 0

    in_channels = 2
    out_channels = 1
    depth = 17
    num_features = 64

    h5_train_path = "train.h5"
    h5_val_path = "val.h5"
    group_name = "hsi_27"

    sigma_json_path = "band_sigmas.json"
    sigma_min = 5.0
    sigma_max = 50.0

    checkpoint_dir = "checkpoints"
    best_model_path = os.path.join(checkpoint_dir, "dncnn_h5_sigma_map_best.pth")
    last_model_path = os.path.join(checkpoint_dir, "dncnn_h5_sigma_map_last.pth")

    # -------------------------
    # Setup
    # -------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)

    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    # -------------------------
    # Data
    # -------------------------
    train_dataset = H5DnCNNDatasetBandSigmaMap(
        h5_path=h5_train_path,
        group_name=group_name,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        normalize=True,
        training=True,
        seed=seed,
        patch_size=64,
        sigma_json_path=sigma_json_path,
        load_sigma_json_if_exists=True,
    )

    val_dataset = H5DnCNNDatasetBandSigmaMap(
        h5_path=h5_val_path,
        group_name=group_name,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        normalize=True,
        training=False,
        seed=seed,
        patch_size=64,
        sigma_json_path=sigma_json_path,
        load_sigma_json_if_exists=True,
    )

    sample = train_dataset[0]
    print("Sample check:")
    print(f'  noisy:      {sample["noisy"].shape}')
    print(f'  sigma_map:  {sample["sigma_map"].shape}')
    print(f'  noise:      {sample["noise"].shape}')
    print(f'  clean:      {sample["clean"].shape}')
    print(f'  band_idx:   {sample["band_idx"]}')
    print(f'  sigma_raw:  {sample["sigma_raw"].item()}')
    print(f'  sigma_norm: {sample["sigma"].item()}')
    print(f'  sigma json: {sigma_json_path}')

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

    # -------------------------
    # Model
    # -------------------------
    model = DnCNNSigmaMap(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        num_features=num_features,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # -------------------------
    # Training
    # -------------------------
    best_val_loss = float("inf")
    train_history = []
    val_history = []
    lr_history = []

    print("\nStarting training...\n")

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

    np.save(os.path.join(checkpoint_dir, "train_history_h5_sigma_map.npy"), np.array(train_history))
    np.save(os.path.join(checkpoint_dir, "val_history_h5_sigma_map.npy"), np.array(val_history))
    np.save(os.path.join(checkpoint_dir, "lr_history_h5_sigma_map.npy"), np.array(lr_history))

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model path: {best_model_path}")


if __name__ == "__main__":
    main()