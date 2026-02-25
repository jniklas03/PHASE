from phase.main.timeseries import Timeseries
from tests.data import TIMELAPSE

ts = Timeseries.from_directory("test", TIMELAPSE)
ts.populate_timeseries()
ts.preprocess_timeseries()
ts.detect_timeseries_old()

ts.export_images(r"C:\Users\Piotr\Desktop\Uni\7. Semester\Thesis\Saved")
ts.plot_counts(r"C:\Users\Piotr\Desktop\Uni\7. Semester\Thesis\Saved")