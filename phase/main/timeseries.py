from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2 as cv
from scipy.spatial import KDTree

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from matplotlib import animation
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects


from ..helpers.inputs import read_time

from .frame import Frame
from .colony import Colony

@dataclass
class Timeseries:
    """
    represents a timelapse series of images, stored as frame objects.

    attributes
    ----------
    name : str
        name of the timeseries
    frames : list[Frame]
        list of frame objects in chronological order
    fg_masks : list[np.ndarray] | None
        foreground masks for each dish, used for preprocessing
    bg_masks : list[np.ndarray] | None
        background masks for each dish, used for preprocessing

    methods
    -------
    from_directory(cls, name, directory)
        create timeseries by loading images from a directory.

    load_timeseries(directory)
        load image files from a directory into frames.
    populate_timeseries(use_stencil=True)
        detect and crop dishes across frames.
    make_masks(n=5)
        generate fg/bg masks for dishes.
    preprocess_timeseries(use_bg_sep=True, n=5)
        preprocess crops using masks.
    export_debug(root="")
        save crops, preprocessed images, and debug overlays.
    """

    name: str
    frames: list[Frame] = field(default_factory=list)
    fg_masks: list[np.ndarray] | None = None
    bg_masks: list[np.ndarray] | None = None

    @classmethod
    def from_directory(cls: type["Timeseries"], name:str, directory: str | Path):
        """
        alternative constructor to create a timeseries from a directory

        parameters
        ----------
        name : str
            name of the timeseries
        directory : str | Path
            path to directory containing images

        returns
        -------
        Timeseries object
        """
        timeseries = cls(name=name)
        timeseries.load_timeseries(directory)
        return timeseries

    def load_timeseries(self, directory: str | Path):
        """
        load all image files (.jpg, .jpeg, .png) from a directory into frames objects

        parameters
        ----------
        directory : str | Path
            directory containing images
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise TypeError("directory must be a string of directory path (str or Path).")
        
        valid_extensions = {".jpg", ".jpeg", ".png"}
        
        for item in tqdm(sorted(directory.iterdir()), desc="Loading frames"): # sorts all entries from given directory
            if item.is_file() and item.suffix.lower() in valid_extensions:
                timestamp = read_time(item.name)

                frame = Frame(
                    name=item.stem,
                    timestamp=timestamp,
                    image_path=Path(item)
                )

                self.frames.append(frame)
    
    def populate_timeseries(self, use_stencil: bool = True):
        """
        populate dishes in each frame of the timeseries

        parameters
        ----------
        use_stencil : bool, optional
            if true, first frame acts as stencil for cropping (default True)
        """
        if not self.frames:
            raise ValueError("Timeseries has no frames to populate.")

        if use_stencil:
            # populate first frame
            self.frames[0].populate_frame()

            # use first frame as stencil for all frames
            stencils = self.frames[0].dishes

            for frame in tqdm(self.frames, desc="Populating frames"):
                frame.populate_frame_from_crop(stencils)
        else:
            # populate each frame independently
            for frame in tqdm(self.frames, desc="Populating frames"):
                frame.populate_frame()
    
    def make_masks(self, n=5):
        """
        generate foreground and background masks for dishes

        parameters
        ----------
        n : int, optional
            number of initial frames used to compute background masks (default 5).
        """
        # foreground mask from last frame
        fg_masks = []
        for dish in tqdm(self.frames[-1].dishes, desc="Making foreground masks"):
            fg_mask = dish.isolate_fg()
            fg_masks.append(fg_mask)

        # bg mask from first n frames
        frame_groups: dict[int, list[np.ndarray]] = defaultdict(list)

        for frame in tqdm(self.frames[:n], desc="Making background masks"):
            for dish in frame.dishes:
                preprocessed = dish.isolate_bg()
                frame_groups[dish.label].append(preprocessed)

        bg_masks = []

        for label in tqdm(sorted(frame_groups), desc="Aggregating background masks"):
            dishes = frame_groups[label]
            aggregate = np.zeros_like(dishes[0])

            for dish in dishes:
                aggregate = cv.bitwise_or(aggregate, dish)

            bg_masks.append(aggregate)

        self.fg_masks = fg_masks
        self.bg_masks = bg_masks

    def preprocess_timeseries(self, use_bg_mask = True, use_fg_mask = False, use_area_filter = False, n=5):
        """
        preprocess each dish for each frame

        parameters
        ----------
        use_bg_mask : bool, optional
            if true, use background masks for preprocessing (default True)
        use_fg_mask : bool, optional
            if true, use foreground masks for preprocessing (default False)
        use_area_filter : bool, optional
            if true, use area filtering for preprocessing (default False)
        n : int, optional
            number of initial frames to compute fg/bg masks (default 5)
        """
        if use_bg_mask or use_fg_mask:
            self.make_masks(n=n)

        for frame in tqdm(self.frames, desc="Preprocessing frames"):
            for dish in frame.dishes:
                dish.preprocessed = dish.preprocess_dish(
                    fg_mask=self.fg_masks[dish.label] if use_fg_mask else None,
                    bg_mask=self.bg_masks[dish.label] if use_bg_mask else None,
                    use_bg_mask=use_bg_mask,
                    use_fg_mask=use_fg_mask,
                    use_area_filter=use_area_filter
                )

    def init_colonies(self):
        for dish in self.frames[0].dishes:
            blobs = dish.detect_colonies()

            colonies = []
        
            for blob in blobs:
                colony = Colony(
                    centroid=(int(blob.pt[0]), int(blob.pt[1])),
                    radius=int(blob.size / 2),
                    growth_rate=0,
                    state="temp"
                )
                colonies.append(colony)

            dish.colonies = colonies            
            dish.count = len(dish.colonies)

    def detect_timeseries_old(self):
        for frame in tqdm(self.frames, desc="Detecting colonies"):
            for dish in frame.dishes:
                blobs = dish.detect_colonies()

                for blob in blobs:
                    colony = Colony(
                        centroid=(int(blob.pt[0]), int(blob.pt[1])),
                        radius=int(blob.size / 2),
                        growth_rate=0
                    )
                    dish.colonies.append(colony)
                
                frame.count = len(dish.colonies)

    def export_images(self, save_path: str = ""):
        """
        export all images contained in a timeseries

        parameters
        ----------
        root : str
            base directory where images will be saved
        """
        save_path = Path(save_path)

        # directories
        (save_path / "dish_detection").mkdir(parents=True, exist_ok=True)
        (save_path / "preprocessed").mkdir(parents=True, exist_ok=True)
        (save_path / "colonies_old").mkdir(parents=True, exist_ok=True)
        (save_path / "colonies").mkdir(parents=True, exist_ok=True)
        (save_path / "fg_masks").mkdir(parents=True, exist_ok=True)
        (save_path / "bg_masks").mkdir(parents=True, exist_ok=True)

        # debug overlay for first frame
        first_frame = self.frames[0]
        overlay = first_frame.image.copy()

        for dish in first_frame.dishes:
            x, y = dish.centroid
            r = dish.radius

            cv.circle(overlay, (x, y), r, (0, 255, 0), 4)
            cv.circle(overlay, (x, y), 10, (0, 0, 255), -1)
            cv.putText(overlay, str(dish.label), (x - 40, y - 40),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        cv.imwrite(str(save_path / "dish_detection" / f"{first_frame.name}_debug.png"), overlay)

        # crops and preprocessed
        for frame in self.frames:
            for dish in frame.dishes:
                if dish.crop is not None:
                    cv.imwrite(str(save_path / "dish_detection" / f"{frame.name}_dish{dish.label}.png"), dish.crop)
                if dish.preprocessed is not None:
                    cv.imwrite(str(save_path / "preprocessed" / f"{frame.name}_dish{dish.label}.png"), dish.preprocessed)

        for frame in self.frames:
            for dish in frame.dishes:
                if dish.detected is not None:
                    cv.imwrite(str(save_path / "colonies_old" / f"{frame.name}_dish{dish.label}.png"), dish.detected)
                if dish.tracked is not None:
                    cv.imwrite(str(save_path / "colonies" / f"{frame.name}_dish{dish.label}.png"), dish.tracked)

        # fg/bg masks
        if self.fg_masks is not None:
            for label, mask in enumerate(self.fg_masks):
                cv.imwrite(str(save_path / "fg_masks" / f"dish{label}.png"), mask)

        if self.bg_masks is not None:
            for label, mask in enumerate(self.bg_masks):
                cv.imwrite(str(save_path / "bg_masks" / f"dish{label}.png"), mask)
    
    def plot_counts(self, save_path: str = "", file_name: str = "colony_counts.png"):
        assert self.frames, "Timeseries has no frames to plot."

        save_path = Path(save_path)

        (save_path / "plots").mkdir(parents=True, exist_ok=True)

        first_frame = self.frames[0]

        cmap = cm.get_cmap("tab10", len(first_frame.dishes))  
        colors = [cmap(i) for i in range(len(first_frame.dishes))]

        dish_data = {dish.label: {"times": [], "counts": []} for dish in first_frame.dishes}

        for frame in self.frames:
            for dish in frame.dishes:
                count = len(dish.colonies) if dish.colonies is not None else 0
                dish_data[dish.label]["times"].append(frame.timestamp)
                dish_data[dish.label]["counts"].append(count)

        plt.figure(figsize=(10, 6))

        for idx, (label, data) in enumerate(dish_data.items()):
            plt.plot(
                data["times"],
                data["counts"],
                marker='o',
                color=colors[idx],
                label=f"Dish {label+1}"
            )

        plt.xlabel('Time')
        plt.ylabel('Colony Count')
        plt.title('Colony Counts Over Time')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(save_path / "plots" / file_name)
        plt.close()
