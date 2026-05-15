from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
COMPARISON_DIR = OUTPUTS_DIR / "comparison"
FGM_COMPARISON_DIR = OUTPUTS_DIR / "fgm_comparison"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir(model_name):
    return ensure_dir(CONFIGS_DIR / model_name)


def checkpoint_dir(model_name):
    return ensure_dir(CHECKPOINTS_DIR / model_name)


def output_dir(model_name):
    return ensure_dir(OUTPUTS_DIR / model_name)
