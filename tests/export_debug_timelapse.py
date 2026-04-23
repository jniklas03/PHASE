from phase.main.timeseries import Timeseries
from phase.helpers.inputs import Image
from tests.data import TIMELAPSE, SAVE_PATH

import pandas as pd
import os

DIR = SAVE_PATH
path = r"C:\Users\Piotr\Desktop\Uni\7. Semester\Thesis\Captures"



if __name__ == "__main__":
    Image.clear_tmp_dir()
    ts = Timeseries.from_directory("test", path,
        clip_beg=100,
        clip_end=300,
        max_images=100
        )
    ts.generate_dishes_timeseries()
    ts.preprocess_timeseries(
        use_bg_mask=True,
        use_fg_mask=True,
        n=30
        )
    ts.detect_timeseries(association_threshold = 0.7, distance_threshold=200)
    stats = ts.compute_stats()
    pd.DataFrame(stats).to_csv(r"C:\Users\Piotr\Desktop\Uni\7. Semester\Thesis\experimental\PHASE\data\Saved\AceGlu\stats.csv", index=False)
    _ = Timeseries.compute_features(stats, save_path=DIR)
    ts.export_mp4(save_path=DIR, fps=60)
    ts.export_images(DIR)
    plot_params = ts.plot_counts(DIR, names=["AG1", "A1", "G1", "A2", "G2", "AG2"])


    rows = []

    for label, fits in plot_params.items():
        for fit_type, params in fits.items():
            L, k, t0 = params

            rows.append({
                "dish": label,
                "fit_type": fit_type,
                "L": L,
                "k": k,
                "t0": t0
            })

    df = pd.DataFrame(rows)
    params_file_path = os.path.join(DIR, "fit_params.csv")
    df.to_csv(params_file_path, index=False)