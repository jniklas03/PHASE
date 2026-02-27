from phase.main.timeseries import Timeseries
from tests.data import SYNTHETIC, SAVE_PATH

DIR = SAVE_PATH

if __name__ == "__main__":
    ts = Timeseries.from_directory("test", SYNTHETIC)
    ts.preprocess_timeseries(use_bg_mask=False, use_fg_mask=False)
    ts.detect_timeseries()
    ts.plot_counts(DIR)
    ts.export_images(DIR)