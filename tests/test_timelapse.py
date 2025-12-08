from phase import timelapse_pipeline
from test_paths import TIMELAPSE, TIMELAPSE_ACEGLU, SAVE_PATH

ANTIBIOTIKA = r"C:\Users\jakub\Documents\Bachelorarbeit\Resources\Sources\Antibiotika"

timelapse_pipeline(
    source=ANTIBIOTIKA, 
    save_intermediates=True,
    save_path=SAVE_PATH, 
    plot=True,
    use_masks=True,
    fine_buffer=2,
    n_to_stack=3
)