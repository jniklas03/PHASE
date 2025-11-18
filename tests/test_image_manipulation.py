import phase
from test_paths import RAW1, RAW2, RAW3, RAW4, RAW5, RAW6, RAW7, RAW8, SAVE_PATH
import os

SOURCE = RAW7

file_name = os.path.splitext(os.path.basename(SOURCE))[0]

dishes, masks, coordinates, metadata = phase.detect_dishes(
    source=SOURCE,
    save_path=SAVE_PATH,
    file_name=file_name,
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
        file_name=file_name,
        idx=idx+1
    )
    preprocessed.append(preprocessed_img)

for idx, preprocessed_img in enumerate(preprocessed):
    phase.detect_colonies(
        source=preprocessed_img,
        raw_img=dishes[idx],
        save=True,
        save_path=SAVE_PATH,
        file_name=file_name,
        idx=idx+1
    )
