"""Training configuration — matches BTDet paper Table 1."""

TRAIN_CONFIG = {
    # Hardware
    "device"       : 0,        # GPU index, use "cpu" for CPU
    
    # Data
    "n_slices"     : 5,        # middle slices per patient
    "img_size"     : 640,      # YOLO input size
    "test_size"    : 0.2,      # val split ratio
    "random_seed"  : 42,

    # Training
    "epochs"       : 50,
    "batch_size"   : 16,
    "optimizer"    : "SGD",
    "lr0"          : 0.01,
    "momentum"     : 0.937,
    "weight_decay" : 0.0005,
    "warmup_epochs": 3,

    # Augmentation
    "mosaic"       : 1.0,
    "mixup"        : 0.1,
    "degrees"      : 10,
    "hsv_h"        : 0.015,
    "hsv_s"        : 0.7,
    "hsv_v"        : 0.4,

    # Clinical encoder
    "clinical_epochs"  : 20,
    "clinical_lr"      : 1e-3,
    "clinical_wd"      : 1e-4,
    "film_channels"    : 256,

    # Survival head
    "survival_epochs"  : 30,
    "survival_lr"      : 1e-3,
    "survival_hidden"  : 128,
}
