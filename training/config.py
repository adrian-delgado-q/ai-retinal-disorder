from __future__ import annotations

from pathlib import Path

from app_config import TrainingSettings, load_config


TOP5_CLASSES = (
    "diabetic_retinopathy",
    "glaucoma",
    "healthy",
    "myopia",
    "macular_scar",
)

SUPPORTED_MODELS = {
    "efficientnet_b0": "efficientnet_b0",
    "resnet18": "resnet18",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
}

DEFAULT_MODEL_NAME = "efficientnet_b0"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 15
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 2
DEFAULT_PATIENCE = 4
DEFAULT_SEED = 42

DEFAULT_TRAIN_CSV = Path("data/splits/train_clean.csv")
DEFAULT_VAL_CSV = Path("data/splits/val_clean.csv")
DEFAULT_TEST_CSV = Path("data/splits/test_clean.csv")

PROVISIONAL_TRAIN_CSV = Path("data/splits/train.csv")
PROVISIONAL_VAL_CSV = Path("data/splits/val.csv")
PROVISIONAL_TEST_CSV = Path("data/splits/test.csv")

REQUIRED_COLUMNS = {
    "image_id",
    "file_path",
    "class_name",
    "split",
    "review_status",
    "image_quality",
    "modality_check",
}

APPROVED_REVIEW_STATUS = "approved"
APPROVED_MODALITY = "fundus"
APPROVED_IMAGE_QUALITIES = {"good", "usable"}


def get_training_settings() -> TrainingSettings:
    return load_config().training
