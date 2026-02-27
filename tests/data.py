from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"

TIMELAPSE = DATA_DIR / "Timelapse" / "T1"
SYNTHETIC = DATA_DIR / "Syntetic"

SAVE_PATH = THIS_DIR / "temp_save"