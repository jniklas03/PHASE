import phase
from test_paths import *
import os

dishes, masks, coordinates, metadata = phase.detect_dishes(
    source=RAW5,
    save_path=SAVE_PATH,
    file_name=os.path.splitext(os.path.basename(RAW5))[0],
    debug=False,
    save=False
)

preprocessed = []

for idx, (dish, mask) in enumerate(zip(dishes, masks)):
    preprocessed_img = phase.preprocess(
        source=dish,
        mask=mask,
        area_filter=False,
        save=True,
        save_path=SAVE_PATH,
        file_name=os.path.splitext(os.path.basename(RAW5))[0],
        idx=idx+1
    )
    preprocessed.append(preprocessed_img)

for idx, preprocessed_img in enumerate(preprocessed):
    phase.detect_colonies(
        source=preprocessed_img,
        raw_img=dishes[idx],
        save=True,
        save_path=SAVE_PATH,
        file_name=os.path.splitext(os.path.basename(RAW5))[0],
        idx=idx
    )
