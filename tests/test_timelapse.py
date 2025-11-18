from phase import timelapse_pipeline
from test_paths import TIMELAPSE, TIMELAPSE_ACEGLU, SAVE_PATH

timelapse_pipeline(
    source=TIMELAPSE_ACEGLU, 
    save_intermediates=True,
    save_path=SAVE_PATH, 
    plot=True
)