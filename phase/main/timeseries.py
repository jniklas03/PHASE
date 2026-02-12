from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2 as cv

from ..helpers.inputs import read_time

from .frame import Frame

@dataclass
class Timeseries:
    name: str
    frames: list[Frame] = field(default_factory=list)
    fg_masks: list[np.ndarray] | None = None
    bg_masks: list[np.ndarray] | None = None

    @classmethod # alternative constructor
    def from_directory(cls: type["Timeseries"], name:str, directory: str | Path):
        timeseries = cls(name=name)
        timeseries.load_timeseries(directory)
        return timeseries

    def load_timeseries(self, directory: str | Path):
        """
        Loads all image files (.jpg, .jpeg, .png) from a given directory into a Timeseries object.

        Parameters
        ----------
        directory: str | Path
            String, or a Path object of a directory containing images.

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
    
    def crop_timeseries(self):
        # generating the dishes at t=0 for crop stencils
        self.frames[0].generate_dishes_dishes()

        stencils = []
        for dish in self.frames[0].dishes:
            stencils.append(dish)

        for frame in self.frames:
            frame.generate_dishes_from_crop(frame, stencils)
    
    def _make_masks(self, n=5):
        # foreground mask
        fg_masks = []
        for dish in self.frames[-1].dishes:
            fg_mask = dish.isolate_fg()
            fg_masks.append(fg_mask)

        # bg mask
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

    def preprocess_timeseries(self, use_bg_sep = True, n=5):
        if use_bg_sep:
            self._makemasks(n=n)

        for frame in self.frames:
            for dish in frame.dishes:
                dish.preprocess(
                    fg_mask=self.fg_masks[dish.label] if use_bg_sep else None,
                    bg_mask=self.bg_masks[dish.label] if use_bg_sep else None,
                    use_bg_sep=use_bg_sep,
                )
