from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


from .timeseries import Timeseries
from .colony import CostFunction

@dataclass
class Run:
    name: str
    timeseries_list: list[Timeseries] = field(default_factory=list)

    @classmethod
    def from_directory(
        cls: type["Run"],
        name:str,
        directory:str | Path,
        load_timeseries=True
        ):
        run = cls(name=name)
        run.load_run(directory, load_timeseries=load_timeseries)
        return run

    def load_run(
            self,
            directory: str | Path,
            load_timeseries=True,
        ):
        directory = Path(directory)

        if not directory.is_dir():
            raise TypeError("directory must be a string of directory path (str or Path).")
        
        for item in tqdm(sorted(directory.iterdir()), desc="Loading run"):
            if item.is_dir():
                timeseries = Timeseries(
                    name=item.name
                )
                
                if load_timeseries:
                    timeseries.load_timeseries(item)

                self.timeseries_list.append(timeseries)

    def populate_run(self, use_stencil=True):
        with ThreadPoolExecutor() as ex:
            list(tqdm(
                ex.map(lambda ts: ts.generate_dishes_timeseries(use_stencil), self.timeseries_list),
                total=len(self.timeseries_list),
                desc="Populating run"
            ))

    def preprocess_run(
            self,
            use_bg_mask = True,
            use_fg_mask = True,
            use_area_filter = False
    ):
        for ts in tqdm(self.timeseries_list, desc="Preprocessing timeseries"):
            ts.preprocess_timeseries(use_bg_mask, use_fg_mask, use_area_filter)

    def detect_run(
            self,
            threshold = 0.9,
            verbosity = 0,
            cost_function: CostFunction = CostFunction.IOU_CIRCLE,
            min_lost_radius = 1
    ):
        for ts in tqdm(self.timeseries_list, desc="Detecting timeseries"):
            ts.detect_timeseries(threshold, verbosity, cost_function, min_lost_radius)

    def export_run(
            self,
            save_path: str | Path = ""
    ):
        save_path = Path(save_path)
        for ts in tqdm(self.timeseries_list, desc="Exporting run"):
            ts_path = save_path / ts.name
            ts_path.mkdir(parents=True, exist_ok=True)
            ts.export_images(ts_path)