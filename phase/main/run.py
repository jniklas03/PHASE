from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os

from .timeseries import Timeseries
from .colony import CostFunction

@dataclass
class Run:
    name: str
    stats: list = field(default_factory=list)

    @classmethod
    def from_directory(
        cls: type["Run"],
        name:str,
        directory:str | Path,
        load_timeseries=True
        ):
        run = cls(name=name)
        run.execute_run(directory, load_timeseries=load_timeseries)
        return run

    def execute_run(
            self,
            directory: str | Path,
            use_stencil=True,
            use_bg_mask=True,
            use_fg_mask=True,
            use_area_filter=False,
            detection_threshold=0.99,
            distance_threshold=10,
            min_lost_radius=2,
            cost_function: CostFunction = CostFunction.IOU_CIRCLE,
            verbosity=0,
            save_path: str | Path = ""
        ):
        directory = Path(directory)

        if not directory.is_dir():
            raise TypeError("directory must be a string of directory path (str or Path).")
        
        for item in tqdm(sorted(directory.iterdir()), desc="Executing run"):
            if item.is_dir():
                save_dir = (save_path / item.name)
                save_dir.mkdir(parents=True, exist_ok=True)
                ts = Timeseries.from_directory(f"{item.name}", item)
                ts.generate_dishes_timeseries(use_stencil=use_stencil)
                ts.preprocess_timeseries(use_bg_mask=use_bg_mask, use_fg_mask=use_fg_mask, use_area_filter=use_area_filter)
                ts.detect_timeseries(detection_threshold=detection_threshold, distance_threshold=distance_threshold, min_lost_radius=min_lost_radius, cost_function=cost_function, verbosity=verbosity)

                stats = ts.export_stats()
                pd.DataFrame(stats).to_csv(os.path.join(save_dir, f"{item.name}_stats.csv"))
                self.stats.extend(stats)

                ts.export_images(save_path=save_dir)
                ts.plot_counts(save_path=save_dir)
                ts.delete()
                del ts
        

