from phase.main.run import Run
from tests.data import SAVE_PATH

DIR = r"C:\Users\Piotr\Desktop\Uni\7. Semester\Thesis\rpoS"
SAVE_DIR = SAVE_PATH

if __name__ == "__main__":
    r1 = Run("r1")
    r1.execute_run(
        directory=DIR,
        save_path=SAVE_DIR,
        clip=80,
        clamp=120,
        detection_threshold=0.6
    )