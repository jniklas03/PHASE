# PHASE: Phenotypic Analysis of Starvation Events
This repo contains an image-processing and tracking pipeline written in Python, based on openCV, developed to analyse colony growth dynamics in carbon starved _E. coli_.
<p align="center">
  <img src="https://github.com/user-attachments/assets/64c29749-6281-4476-bc31-5441be814f5b" height="500"/>
</p>

## Theory

_E. coli_ in carbon starvation can exhibit death dynamics resembling those of heterogenous subpopulations (biphasic instead of monophasic).
<p align="center">
  <img src="https://github.com/user-attachments/assets/e7c9f95e-c91c-4936-b38c-5749797d348c" height="300"/>
</p>
In order to analyse these characteristics further, regrowth dynamics were analysed. I.e. starving bacteria were reintroduced to nutrients, imaged, and features were extracted.
This allowed for single colony resolution, which prevented averaging over potentially heterogenous subpopulations.
<p align="center">
  <img src="https://github.com/user-attachments/assets/2ee77b56-ee07-4b65-b012-cb012990a393" height="300" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/a9b8093e-b772-4178-b61c-4da8ed287cd5" height="300"/>
</p>


## Pipeline overview
The pipeline is centered around Kalman-filter-based tracking instead of naive per-frame detection in order to better deal with noise and colony merges. Additionally, it takes advantage of biological constraints and uses greedy spatial matching via KDTrees instead of exhaustive matching. To further stay performant, the parallelisation is employed for most of its processing steps.
<p align="center">
  <img src="https://github.com/user-attachments/assets/10c42164-bd70-4255-bf8f-8bec04c54e7f" height="300"/>
</p>

1. **Dish Detection & Cropping**
   - Tracking was to be done on separate dishes in order to isolate technical replicates
   - Strong blurring and CLAHE was applied in order to only preserve dish contours
   - Dish detection via Hough Transform

2. **Preprocessing**
   - Fore- and background separation for artefact removal
   - Adaptive thresholding

3. **Colony Tracking**
   - Colony detection via blob detector
   - Kalman filter-based tracking
   - Greedy matching using KDTrees
   - Handling of new/lost/merged colonies

5. **Feature Extraction**
   - Lag time
   - Colony radius
   - Expansion rate

## Installation
```bash
pip install git+https://github.com/jniklas03/PHASE/
```
## Usage
Test data is provided via `phase.data.TIMELAPSE`. Exemplary usage in processing a multi-day experiment below.


```python
from phase.main.timeseries import Timeseries
from phase.helpers.inputs import Image, read_image_paths

import pandas as pd
import os
from pathlib import Path

SAVE = r"foo"
READ = r"bar"

days = os.listdir(READ)

# iterating over a directory with multiple days worth of experiments
for day in days:
    curr_save = Path(os.path.join(SAVE, day))
    curr_read = Path(os.path.join(READ, day))
    os.makedirs(curr_save, exist_ok=True)

    # when processing multi-day data, i.e. building multiple timeseries, clearing the cache is recommended
    Image.clear_tmp_dir()

    # creating a timeseries from directory, for each given day
    ts = Timeseries.from_directory(
        f"{day}",
        curr_read,
        clip_end=6  # leaving out the last 6 frames, due to contamination
    )

    ts.generate_dishes_timeseries()

    # preprocessing the timeseries using bg- and fg-separation
    ts.preprocess_timeseries(
        use_bg_mask=True,
        use_fg_mask=True,
        )
    
    ts.detect_timeseries(association_threshold = 0.5, distance_threshold=10)

    # extracting final counts per day for further analysis
    counts = []
    for i, dish in enumerate(ts.frames[-1].dishes):
        counts.append(dish.count)
        with open(os.path.join(curr_save, f"counts_{day}.txt"), "w") as f:
            for c in counts:
                f.write(f"{c}\n")

    
    # computing and saving stats as well as compound features
    stats = ts.compute_stats()
    stats_path = os.path.join(curr_save, f"stats_{day}.csv")
    pd.DataFrame(stats).to_csv(stats_path, index=False)
    _ = Timeseries.compute_features(stats, save_path=curr_save)

    # exporting images, videos, and plots
    ts.export_mp4(save_path=curr_save, fps=60)
    ts.export_images(curr_save)
    plot_params = ts.plot_counts(curr_save, names=["WT1R1", "WT1R2", "WT1R3", "WT2R1", "WT2R2", "WT2R3"])

    # exporting fit parameters of plots
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
    params_file_path = os.path.join(curr_save, f"fit_params_{day}.csv")
    df.to_csv(params_file_path, index=False)
```
