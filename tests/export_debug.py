from phase.main.timeseries import Timeseries
from tests.data import TIMELAPSE

ts = Timeseries.from_directory("test", TIMELAPSE)
ts.populate_timeseries()
ts.preprocess_timeseries()

ts.export_debug(r"C:\Users\Piotr\Desktop\Uni\7. Semester\Thesis\Saved")