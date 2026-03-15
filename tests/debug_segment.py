from phase.main.timeseries import Timeseries
from phase.main.dish import Dish

from tests.data import TIMELAPSE, SAVE_PATH

DIR = SAVE_PATH

if __name__ == "__main__":
    ts = Timeseries.from_directory("test", TIMELAPSE)
    ts.generate_dishes_timeseries()
    ts.preprocess_timeseries()
    ts.frames[-1].dishes[0].segment(DIR)