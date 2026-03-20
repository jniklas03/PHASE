from phase.main.timeseries import Timeseries
from tests.data import TIMELAPSE, SAVE_PATH

DIR = SAVE_PATH

if __name__ == "__main__":
    ts = Timeseries.from_directory("test", TIMELAPSE)
    ts.generate_dishes_timeseries()
    ts.preprocess_timeseries()
    ts.detect_timeseries(detection_threshold = 0.6)
    ts.export_gif(save_path=DIR, fps=60)
    ts.plot_counts(DIR)
    ts.export_images(DIR)