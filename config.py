from pathlib import Path

_ROOT = Path(__file__).parent

PTBXL_PATH     = _ROOT / "data" / "ptbxl"
HELME_DATA     = _ROOT / "output" / "ecg_ptbxl_benchmarking" / "exp0" / "data"
CHECKPOINT_DIR = str(_ROOT / "checkpoints")
SAMPLING_RATE  = 100
BATCH_SIZE     = 64
EPOCHS         = 30
LR             = 1e-3
NUM_CLASSES    = 5   # superdiagnostic: NORM, MI, STTC, CD, HYP
