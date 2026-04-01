import os


class Config:
    # =========================
    # Paths
    # =========================
    DATA_DIR = "Data"
    TRAIN_H5 = os.path.join(DATA_DIR, "train.h5")
    VAL_H5 = os.path.join(DATA_DIR, "val.h5")
    TEST_H5 = os.path.join(DATA_DIR, "test.h5")

    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"

    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "dncnn_best.pth")
    LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "dncnn_last.pth")

    # =========================
    # Reproducibility
    # =========================
    SEED = 42

    # =========================
    # Model
    # =========================
    IN_CHANNELS = 1         # 1 for grayscale, 3 for RGB
    DEPTH = 17
    NUM_FEATURES = 64
    KERNEL_SIZE = 3

    # =========================
    # Noise
    # =========================
    SIGMA = 25              # Gaussian noise level

    # =========================
    # Training
    # =========================
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 0

    # =========================
    # Scheduler
    # =========================
    STEP_SIZE = 20
    GAMMA = 0.5

    # =========================
    # Testing
    # =========================
    TEST_BATCH_SIZE = 1
    SAVE_OUTPUTS = True

    # =========================
    # Data settings
    # =========================
    NORMALIZE = True