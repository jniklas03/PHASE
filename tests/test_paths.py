from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"

RAW1 = DATA_DIR / "Singles" / "02.10.2025-04.30.02.jpg"
RAW2 = DATA_DIR / "Singles" / "28.10.2025-02.30.02.jpg"
RAW3 = DATA_DIR / "Singles" / "30.10.2025-01.00.02.jpg"
RAW4 = DATA_DIR / "Singles" / "30.10.2025-09.30.02.jpg"
RAW5 = DATA_DIR / "Singles" / "Acetate-Glucose" / "05.11.2025-13.30.02.jpg"
RAW6 = DATA_DIR / "Singles" / "Acetate-Glucose" / "05.11.2025-22.00.02.jpg"
RAW7 = DATA_DIR / "Singles" / "Acetate-Glucose" / "07.11.2025-09.00.02.jpg"
RAW8 = DATA_DIR / "Singles" / "Acetate-Glucose" / "09.11.2025-22.00.02.jpg"

TIMELAPSE = DATA_DIR / "Timelapse" / "29.10.2025"
TIMELAPSE_ACEGLU = DATA_DIR / "Timelapse" / "AceGlu"

SAVE_PATH = DATA_DIR / "Save"