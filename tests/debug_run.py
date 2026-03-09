from phase.main.run import Run
from tests.data import SAVE_PATH

DIR = r"C:\Users\Piotr\Desktop\Uni\7. Semester\Thesis\rpoS"
SAVE_DIR = SAVE_PATH

if __name__ == "__main__":
    r1 = Run.from_directory("long_starve", DIR)
    r1.populate_run()
    r1.preprocess_run()
    r1.detect_run()
    r1.export_run(SAVE_DIR)