from phase import timelapse_pipeline
from test_paths import *

timelapse_pipeline(
    source=TIMELAPSE, 
    save_intermediates=True,
    save_path=SAVE_PATH, 
    plot=True
)