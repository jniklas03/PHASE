from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2 as cv
from scipy.spatial import KDTree

import matplotlib.pyplot as plt
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
        
        for item in sorted(directory.iterdir()): # sorts all entries from given directory
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

            for frame in self.frames:
                frame.populate_frame_from_crop(stencils)
        else:
            # populate each frame independently
            for frame in self.frames:
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
        for dish in self.frames[-1].dishes:
            fg_mask = dish.isolate_fg()
            fg_masks.append(fg_mask)

        # bg mask from first n frames
        frame_groups: dict[int, list[np.ndarray]] = defaultdict(list)

        for frame in self.frames[:n]:
            for dish in frame.dishes:
                preprocessed = dish.isolate_bg()
                frame_groups[dish.label].append(preprocessed)

        bg_masks = []

        for label in sorted(frame_groups):
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

        for frame in self.frames:
            for dish in frame.dishes:
                dish.preprocessed = dish.preprocess_dish(
                    fg_mask=self.fg_masks[dish.label] if use_fg_mask else None,
                    bg_mask=self.bg_masks[dish.label] if use_bg_mask else None,
                    use_bg_mask=use_bg_mask,
                    use_fg_mask=use_fg_mask,
                    use_area_filter=use_area_filter
                )
                
    def export_debug(self, root: str = ""):
        """
        export all images contained in a timeseries

        parameters
        ----------
        root : str
            base directory where images will be saved
        """
        root = Path(root)

        # directories
        (root / "dish_detection").mkdir(parents=True, exist_ok=True)
        (root / "preprocessed").mkdir(parents=True, exist_ok=True)
        (root / "fg_masks").mkdir(parents=True, exist_ok=True)
        (root / "bg_masks").mkdir(parents=True, exist_ok=True)

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

        cv.imwrite(str(root / "dish_detection" / f"{first_frame.name}_debug.png"), overlay)

        # crops and preprocessed
        for frame in self.frames:
            for dish in frame.dishes:
                if dish.crop is not None:
                    cv.imwrite(str(root / "dish_detection" / f"{frame.name}_dish{dish.label}.png"), dish.crop)
                if dish.preprocessed is not None:
                    cv.imwrite(str(root / "preprocessed" / f"{frame.name}_dish{dish.label}.png"), dish.preprocessed)

        # fg/bg masks
        if self.fg_masks is not None:
            for label, mask in enumerate(self.fg_masks):
                cv.imwrite(str(root / "fg_masks" / f"dish{label}.png"), mask)

        if self.bg_masks is not None:
            for label, mask in enumerate(self.bg_masks):
                cv.imwrite(str(root / "bg_masks" / f"dish{label}.png"), mask)