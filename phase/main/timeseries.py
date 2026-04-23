from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import KDTree
from itertools import repeat
import os
import pandas as pd
import imageio.v2 as imageio
import re
import scienceplots  # dont remove, needed for plot formatting!

from .frame import Frame
from .colony import Colony, CostFunction
from .dish import Dish

from ..helpers.inputs import read_time, Image, read_image_paths


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
    next_label: int = 0

    @classmethod
    def from_directory(
        cls: type["Timeseries"],
        name: str,
        directory: str | Path,
        clip_beg: int | float | None = None,
        clip_end: int | float | None = None,
        max_images: int | None = None,
        sample_fraction: float | None = None,
    ) -> None:
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
        timeseries.load_timeseries(
            directory, clip_beg, clip_end, max_images, sample_fraction
        )
        return timeseries

    def load_timeseries(
        self,
        directory: str | Path,
        clip_beg: int | float | None = None,
        clip_end: int | float | None = None,
        max_images: int | None = None,
        sample_fraction: float | None = None,
    ) -> None:
        """
        method for loading images into timeseries object

        Args:
            directory (str | Path): _Directory where the images are stored._
            clip_beg (int | float | None, optional): _Amount or fraction of images in the beginning to ignore._ Defaults to None.
            clip_end (int | float | None, optional): _Amount or fraction of images in the end to ignore._ Defaults to None.
            max_images (int | None, optional): Amount of images to be uniformly sampled from the directory. Defaults to None.
            sample_fraction (float | None, optional): _Fraction of images to be uniformly sampled from the directory._ Defaults to None.

        Raises:
            TypeError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """

        directory = Path(directory)
        if not directory.is_dir():
            raise TypeError(
                "directory must be a string of directory path (str or Path)."
            )

        if sample_fraction is not None and not (0 < sample_fraction <= 1):
            raise ValueError("fraction must be between 0 and 1.")

        if sum(x is not None for x in [max_images, sample_fraction]) > 1:
            raise ValueError("Only one of max_images or sample_fraction can be set.")

        items, _ = read_image_paths(directory)

        if clip_beg is not None:
            if isinstance(clip_beg, int):
                if clip_beg < 0:
                    raise ValueError("clip (int) must be >= 0.")
                items = items[clip_beg:]
            elif isinstance(clip_beg, float):
                if not (0 <= clip_beg <= 1):
                    raise ValueError("clip (float) must be between 0 and 1.")
                n = int(len(items) * clip_beg)
                items = items[n:]
            else:
                raise TypeError("clip must be int or float.")

        if clip_end is not None:
            if isinstance(clip_end, int):
                if clip_end <= 0:
                    raise ValueError("clamp (int) must be > 0.")

                if clip_end >= len(items):
                    items = []
                else:
                    items = items[:-clip_end]

            elif isinstance(clip_end, float):
                if not (0 < clip_end <= 1):
                    raise ValueError("clamp (float) must be between 0 and 1.")

                n_remove = int(len(items) * clip_end)

                if n_remove >= len(items):
                    items = []
                else:
                    items = items[:-n_remove]

        if max_images is not None and sample_fraction is not None:
            raise ValueError("Only one of max_images or sample_fraction can be set.")

        if max_images is not None:
            indices = np.linspace(0, len(items) - 1, max_images, dtype=int)
            items = [items[i] for i in indices]

        elif sample_fraction is not None:
            n_to_load = max(1, int(len(items) * sample_fraction))
            indices = np.linspace(0, len(items) - 1, n_to_load, dtype=int)
            items = [items[i] for i in indices]

        # Final result
        selected_items = items

        for item in tqdm(selected_items, desc="Loading frames"):
            timestamp = read_time(item.name)
            frame = Frame(name=item.stem, timestamp=timestamp, image=Image(item))
            self.frames.append(frame)

    def generate_dishes_timeseries(self, use_stencil: bool = True) -> None:
        """
        Generates dish objects from a timeseries object.

        Args:
            use_stencil (bool, optional): _Whether to use the first frame as a cropping stencil for all the frames. Disable if the recording was shaky._ Defaults to True.

        Raises:
            ValueError: _description_
        """
        if not self.frames:
            raise ValueError("Timeseries has no frames to populate.")

        def _populate_frame(frame, stencils):
            frame.populate_frame_from_crop(stencils)
            return frame

        def _populate_frame_no_crop(frame):
            frame.populate_frame()
            return _populate_frame

        if use_stencil:
            # populate first frame
            self.frames[0].populate_frame()

            # use first frame as stencil for all frames
            stencils = self.frames[0].dishes

            with ThreadPoolExecutor() as ex:
                self.frames = list(
                    tqdm(
                        ex.map(lambda f: _populate_frame(f, stencils), self.frames[1:]),
                        total=len(self.frames[1:]),
                        desc="Generating dishes",
                    )
                )

        else:
            # populate each frame independently
            with ThreadPoolExecutor() as ex:
                self.frames = list(
                    tqdm(
                        ex.map(lambda f: _populate_frame_no_crop(), self.frames),
                        total=len(self.frames),
                        desc="Generating dishes",
                    )
                )

    def make_masks(self, n=5) -> None:
        """
        Creates fore- and background masks for the timeseries.

        Args:
            n (int, optional): _Number of initial frames used to compute background masks._ Defaults to 5.
        """

        # foreground mask from last frame
        def _make_fg_mask(dish):
            return dish.label, dish.isolate_fg()

        with ThreadPoolExecutor() as ex:
            results = list(
                tqdm(
                    ex.map(_make_fg_mask, self.frames[-1].dishes),
                    total=len(self.frames[-1].dishes),
                    desc="Making foreground masks",
                )
            )

        fg_masks = [None] * len(results)
        for label, mask in results:
            fg_masks[label] = mask

        # bg mask from first n frames
        def _make_bg_mask(dish):
            return dish.label, dish.isolate_bg()

        frame_groups: dict[int, list[np.ndarray]] = defaultdict(list)

        dishes = [dish for frame in self.frames[:n] for dish in frame.dishes]

        with ThreadPoolExecutor() as ex:
            results = list(
                tqdm(
                    ex.map(_make_bg_mask, dishes),
                    total=len(dishes),
                    desc="Making background masks",
                )
            )

        for label, mask in results:
            frame_groups[label].append(mask)

        bg_masks = []

        for label in tqdm(sorted(frame_groups), desc="Aggregating background masks"):
            dishes = frame_groups[label]
            aggregate = np.zeros_like(dishes[0])

            for dish in dishes:
                aggregate = cv.bitwise_or(aggregate, dish)

            bg_masks.append(aggregate)

        self.fg_masks = fg_masks
        self.bg_masks = bg_masks

    def get_new_label(self) -> int:
        label = self.next_label
        self.next_label += 1
        return label

    def preprocess_timeseries(
        self, use_bg_mask=True, use_fg_mask=True, use_area_filter=False, n=5
    ) -> None:
        """
        Applies preprocessing to the timeseries.

        Args:
            use_bg_mask (bool, optional): _Whether or not to use background masking_. Defaults to True.
            use_fg_mask (bool, optional): _Whether or not to use foreground masking_. Defaults to True.
            use_area_filter (bool, optional): _Whether or not to filter large elements; deprecated, keep off_. Defaults to False.
            n (int, optional): _Amount of frames to average for the background mask_. Defaults to 5.
        """
        if use_bg_mask or use_fg_mask:
            self.make_masks(n=n)

        def _preprocess_frame(frame):
            for dish in frame.dishes:
                dish.preprocessed = Image(
                    dish.preprocess_dish(
                        fg_mask=self.fg_masks[dish.label] if use_fg_mask else None,
                        bg_mask=self.bg_masks[dish.label] if use_bg_mask else None,
                        use_bg_mask=use_bg_mask,
                        use_fg_mask=use_fg_mask,
                        use_area_filter=use_area_filter,
                    )
                )
            return frame

        with ThreadPoolExecutor() as ex:
            list(
                tqdm(
                    ex.map(_preprocess_frame, self.frames),
                    total=len(self.frames),
                    desc="Preprocessing frames",
                )
            )

    def detect_timeseries_old(self):
        """
        Deprecated detection method for timeseries. Just uses naive blob detection.
        """
        for frame in tqdm(self.frames, desc="Detecting colonies"):
            for dish in frame.dishes:
                blobs = dish.detect_colonies()

                for blob in blobs:
                    colony = Colony(
                        centroid=(int(blob.pt[0]), int(blob.pt[1])),
                        radius=int(blob.size / 2),
                        expansion_rate=0,
                        label=self.get_new_label(),
                    )
                    dish.colonies.append(colony)

                frame.count = len(dish.colonies)

    def detect_timeseries(
        self,
        distance_threshold: int = 10,
        association_threshold: float = 0.5,
        min_lost_radius: int = 2,
        cost_function: CostFunction = CostFunction.IOU_CIRCLE,
        verbosity: int = 0,
    ) -> None:
        """
        Detects colonies in timeseries with temporal inference.

        Args:
            distance_threshold (int, optional): _Threshold for kdtree query. Colonies beyond that radius won't be considered for linking_. Defaults to 10.
            association_threshold (float, optional): _Scoring threshold to link two colonies temporally_. Defaults to 0.5.
            min_lost_radius (int, optional): _Minimal radius required to keep lost colonies, prevents noise_. Defaults to 2.
            cost_function (CostFunction, optional): _Used cost function_. Defaults to CostFunction.IOU_CIRCLE.
            verbosity (int, optional): _Verbosity for debugging drawing the colonies, don't use_. Defaults to 0.
        """

        # 0. grab dt for kalman filter (normalised for 30 mins)
        dt = (
            self.frames[1].timestamp - self.frames[0].timestamp
        ).total_seconds() / 1800.0

        # 1. init first frame colonies
        for dish in self.frames[0].dishes:
            blobs = dish.detect_colonies()

            for blob in blobs:
                dish.colonies.append(
                    Colony(
                        centroid=(int(blob.pt[0]), int(blob.pt[1])),
                        radius=float(blob.size / 2),
                        expansion_rate=0,
                        label=self.get_new_label(),  # getting unique label from timeseries
                        state="temp",
                        age=1,
                    )
                )

        def _detect_worker(
            prev_dish,
            curr_dish,
            dt,
            distance_threshold=distance_threshold,
            association_threshold=association_threshold,
            min_lost_radius=min_lost_radius,
            cost_function=cost_function,
            verbosity=verbosity,
        ) -> Dish:
            """
            Worker function used for parallelisation.
            """

            # 2. init prev and current colonies for dish pairs
            prev_cols = prev_dish.colonies
            curr_dish.colonies = []

            # 3. predict colony states
            predicted_cols = [c.predict(dt) for c in prev_cols]

            # 4. apply growth extrapolation to previously lost colonies, mask them, and detect colonies for current dish
            lost_mask = np.zeros_like(prev_dish.preprocessed.load())

            for col in predicted_cols:
                if col.state == "lost":
                    r = max(int(col.radius), 1)  # failsafe: minimum radius of 1
                    x, y = int(col.centroid[0]), int(col.centroid[1])

                    cv.circle(lost_mask, (x, y), r, 255, -1)

            preprocessed_masked = cv.bitwise_and(
                curr_dish.preprocessed.load(), cv.bitwise_not(lost_mask)
            )

            detected_blobs, _ = Dish.colony_detection(
                preprocessed_masked, curr_dish.crop.load()
            )

            curr_dish.preprocessed_masked = Image(preprocessed_masked)

            # 5. building trees and assigning candidates
            # 5.1 make candidate pairs using KDTree (prev_blobs x curr_blobs) (replacement of hungarian)
            n, m = len(predicted_cols), len(detected_blobs)
            matches = []

            if n > 0 and m > 0:
                det_centroids = np.array([b.pt for b in detected_blobs])

                tree = KDTree(det_centroids)

                candidate_pairs = []

                # distance gating
                for i, pred_col in enumerate(predicted_cols):
                    neighbors = tree.query_ball_point(
                        pred_col.centroid, distance_threshold
                    )

                    for j in neighbors:
                        cost = cost_function(pred_col, detected_blobs[j])

                        if cost < association_threshold:
                            candidate_pairs.append((cost, i, j))

                # sorting (lowest cost first)
                candidate_pairs.sort()

                used_pred = set()
                used_det = set()

                for cost, i, j in candidate_pairs:
                    if i not in used_pred and j not in used_det:
                        matches.append((i, j))
                        used_pred.add(i)
                        used_det.add(j)

            # 5.2. assignment
            matched_pred = [predicted_cols[r] for r, _ in matches]
            matched_det = [detected_blobs[c] for _, c in matches]

            matched_pred_idx = {r for r, _ in matches}
            matched_det_idx = {c for _, c in matches}

            unmatched_pred = [
                predicted_cols[i] for i in range(n) if i not in matched_pred_idx
            ]  # colonies that disappeared
            unmatched_det = [
                detected_blobs[j] for j in range(m) if j not in matched_det_idx
            ]  # colonies that newly appeared

            # 6. handling 3 possible states and updating kalman
            # 6.1. link matched colonies
            for pred_col, det_blob in zip(matched_pred, matched_det):
                measured_radius = float(det_blob.size / 2)
                measured_centroid = det_blob.pt

                # update Kalman filter with new measurement
                pred_col.update(measured_centroid, measured_radius)
                pred_col.missed_frames = 0

                if pred_col.age >= 3:
                    pred_col.state = "perm"

                curr_dish.colonies.append(pred_col)

            # 6.2. newly lost colony handling
            for pred_col in unmatched_pred:
                pred_col.missed_frames += 1
                if pred_col.radius >= min_lost_radius and (
                    pred_col.state == "perm" or pred_col.state == "lost"
                ):
                    pred_col.state = "lost"
                    curr_dish.colonies.append(pred_col)
                else:
                    # deletes track if missing too long
                    pass

            # 6.3. add new colonies
            for blob in unmatched_det:
                new_radius = float(blob.size / 2)
                new_centroid = blob.pt

                colony = Colony(
                    centroid=(int(new_centroid[0]), int(new_centroid[1])),
                    radius=new_radius,
                    label=self.get_new_label(),
                    state="temp",
                    age=1,
                    expansion_rate=0.0,
                    missed_frames=0,
                )

                curr_dish.colonies.append(colony)

            # 7. update dish count and draw tracked colonies
            curr_dish.count = len(curr_dish.colonies)

            curr_dish.draw_tracked_colonies(verbosity=verbosity)

            return curr_dish

        # 8. iteratation over [1:] frames with worker
        with ThreadPoolExecutor() as ex:
            for n in tqdm(range(1, len(self.frames)), desc="Tracking colonies"):
                prev_frame = self.frames[n - 1]
                curr_frame = self.frames[n]

                list(
                    ex.map(
                        _detect_worker,
                        prev_frame.dishes,
                        curr_frame.dishes,
                        repeat(dt),
                        repeat(distance_threshold),
                        repeat(association_threshold),
                        repeat(min_lost_radius),
                        repeat(cost_function),
                        repeat(verbosity),
                    )
                )

    def export_images(self, save_path: str | Path = ""):
        """
        Exports all the images stored within the timeseries.

        Args:
            save_path (str | Path, optional): _Root directory of where the stored images should be saved_. Defaults to "".
        """
        save_path = Path(save_path)

        # directories
        (save_path / "dish_detection").mkdir(parents=True, exist_ok=True)
        (save_path / "preprocessed").mkdir(parents=True, exist_ok=True)
        (save_path / "preprocessed_masked").mkdir(parents=True, exist_ok=True)
        (save_path / "initial_detection").mkdir(parents=True, exist_ok=True)
        (save_path / "tracked_detection").mkdir(parents=True, exist_ok=True)
        (save_path / "fg_masks").mkdir(parents=True, exist_ok=True)
        (save_path / "bg_masks").mkdir(parents=True, exist_ok=True)

        # debug overlay for first frame
        first_frame = self.frames[0]
        overlay = first_frame.image.load().copy()

        for dish in first_frame.dishes:
            x, y = dish.centroid
            r = dish.radius

            cv.circle(overlay, (x, y), r, (0, 255, 0), 4)
            cv.circle(overlay, (x, y), 10, (0, 0, 255), -1)
            cv.putText(
                overlay,
                str(dish.label),
                (x - 40, y - 40),
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                3,
            )

        cv.imwrite(
            str(save_path / "dish_detection" / f"{first_frame.name}_debug.png"), overlay
        )

        def _save_images(frame) -> None:
            """
            Worker function for parallelisation.
            """
            # crops, preprocessed, detections
            for dish in frame.dishes:
                if dish.crop is not None:
                    cv.imwrite(
                        str(
                            save_path
                            / "dish_detection"
                            / f"{frame.name}_dish{dish.label}.png"
                        ),
                        dish.crop.load(),
                    )

                if dish.preprocessed is not None:
                    cv.imwrite(
                        str(
                            save_path
                            / "preprocessed"
                            / f"{frame.name}_dish{dish.label}.png"
                        ),
                        dish.preprocessed.load(),
                    )

                if dish.preprocessed_masked is not None:
                    cv.imwrite(
                        str(
                            save_path
                            / "preprocessed_masked"
                            / f"{frame.name}_dish{dish.label}.png"
                        ),
                        dish.preprocessed_masked.load(),
                    )

                if dish.initial_detection is not None:
                    cv.imwrite(
                        str(
                            save_path
                            / "initial_detection"
                            / f"{frame.name}_dish{dish.label}.png"
                        ),
                        dish.initial_detection.load(),
                    )

                if dish.tracked_detection is not None:
                    cv.imwrite(
                        str(
                            save_path
                            / "tracked_detection"
                            / f"{frame.name}_dish{dish.label}.png"
                        ),
                        dish.tracked_detection.load(),
                    )

        with ThreadPoolExecutor() as ex:
            list(
                tqdm(
                    ex.map(_save_images, self.frames),
                    total=len(self.frames),
                    desc="Saving images",
                )
            )

        # fg/bg masks
        if self.fg_masks is not None:
            for label, mask in enumerate(self.fg_masks):
                cv.imwrite(str(save_path / "fg_masks" / f"dish{label}.png"), mask)

        if self.bg_masks is not None:
            for label, mask in enumerate(self.bg_masks):
                cv.imwrite(str(save_path / "bg_masks" / f"dish{label}.png"), mask)

    def plot_counts(
        self,
        save_path: str | Path = "",
        file_name: str = "colony_counts.png",
        names: list[str] | None = None,
    ) -> dict:
        """
        Automatic plotting of the timeseries data. Plots a) colony counts over time b) normalised colony counts over time c) log 1-normalised colony counts over time.

        Args:
            save_path (str | Path, optional): _Root directory where the plots should be saved_. Defaults to "".
            file_name (str, optional): _Root of file name_. Defaults to "colony_counts.png".
            names (list[str] | None, optional): _Legend names of detected dishes_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            dict: _Fit parameters of the gompertz fits._
        """

        assert self.frames, "Timeseries has no frames to plot."

        save_path = Path(save_path)
        (save_path / "plots").mkdir(parents=True, exist_ok=True)

        first_frame = self.frames[0]
        t0 = first_frame.timestamp

        # apply input names for legend labels
        dish_labels = [dish.label for dish in first_frame.dishes]
        if names is not None:
            if len(names) != len(dish_labels):
                raise ValueError("Length of names must match number of dishes")
            label_map = dict(zip(dish_labels, names))
        else:
            label_map = {label: f"Dish {label + 1}" for label in dish_labels}

        # attempts to sort names like "A1, A2; AG1, AG2; G1, G2" together
        def natural_key(s):
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split(r"(\d+)", str(s))
            ]

        sorted_items = sorted(label_map.items(), key=lambda x: natural_key(x[1]))
        sorted_labels = [label for label, _ in sorted_items]

        # grabs clean data
        dish_data = {label: {"times": [], "counts": []} for label in dish_labels}

        # attempts to apply scienceplots style
        try:
            plt.style.use(["science", "scatter"])
        except Exception:
            plt.style.use(["science", "no-latex", "scatter"])

        # gompertz fit
        def gompertz(t, L, k, t0):
            return L * np.exp(-np.exp(-k * (t - t0)))

        # converts timestamps to delta t
        for frame in self.frames:
            dt = (frame.timestamp - t0).total_seconds() / 3600.0

            for dish in frame.dishes:
                count = sum(
                    1 for col in (dish.colonies or []) if col.state in {"perm", "lost"}
                )
                dish_data[dish.label]["times"].append(dt)
                dish_data[dish.label]["counts"].append(count)

        fit_results = {}

        # RAW PLOT
        plt.figure()
        for label in sorted_labels:
            data = dish_data[label]
            x = np.array(data["times"])
            y = np.array(data["counts"])

            plt.plot(x, y, label=label_map[label], markerfacecolor="none")

            p0 = [max(y), 0.1, np.median(x)]
            params, _ = curve_fit(gompertz, x, y, p0=p0)

            fit_results.setdefault(label, {})["raw"] = tuple(params)

            L, k, t0_fit = params
            t_fit = np.linspace(min(x), max(x), 200)
            plt.plot(
                t_fit,
                gompertz(t_fit, L, k, t0_fit),
                "-",
                color=plt.gca().lines[-1].get_color(),
            )

        plt.xlabel("Time [h]")
        plt.ylabel("Colony count")
        plt.legend(loc=2, prop={"size": 6})
        plt.tight_layout()
        plt.savefig(
            save_path / "plots" / file_name.replace(".png", "_raw.png"), dpi=300
        )
        plt.close()

        # NORMALISED PLOT
        plt.figure()
        for label in sorted_labels:
            data = dish_data[label]
            x = np.array(data["times"])
            y = np.array(data["counts"])
            y_norm = y / max(y)

            plt.plot(x, y_norm, label=label_map[label], markerfacecolor="none")

            p0 = [1.0, 0.1, np.median(x)]
            params, _ = curve_fit(gompertz, x, y_norm, p0=p0)

            fit_results[label]["normalized"] = tuple(params)

            L, k, t0_fit = params
            t_fit = np.linspace(min(x), max(x), 200)
            plt.plot(
                t_fit,
                gompertz(t_fit, L, k, t0_fit),
                "-",
                color=plt.gca().lines[-1].get_color(),
            )

        plt.xlabel("Time [h]")
        plt.ylabel("Normalized colony count")
        plt.legend(loc=2, prop={"size": 6})
        plt.tight_layout()
        plt.savefig(
            save_path / "plots" / file_name.replace(".png", "_normalized.png"), dpi=300
        )
        plt.close()

        # LOG 1-NORM PLOT
        plt.figure()
        for label in sorted_labels:
            data = dish_data[label]
            x = np.array(data["times"])
            y = np.array(data["counts"])
            y_norm = y / max(y)
            y_frac = 1 - y_norm

            mask = y_frac > 0
            x_masked = x[mask]
            y_masked = y_frac[mask]

            plt.plot(x_masked, y_masked, label=label_map[label], markerfacecolor="none")

            p0 = [1.0, 0.1, np.median(x_masked)]
            params, _ = curve_fit(gompertz, x_masked, 1 - y_masked, p0=p0)

            fit_results[label]["log"] = tuple(params)

        plt.xlabel("Time [h]")
        plt.ylabel("log(1 - fraction of max count)")
        plt.yscale("log")
        plt.legend(loc=3, prop={"size": 6})
        plt.tight_layout()
        plt.savefig(
            save_path / "plots" / file_name.replace(".png", "_log1m.png"), dpi=300
        )
        plt.close()

        return fit_results

    def compute_stats(self) -> list:
        """
        _Extracts stats from the timeseries object_

        Returns:
            list: _Timeseries statistics_
        """
        rows = []

        for frame_idx, frame in tqdm(enumerate(self.frames), desc="Exporting data"):
            for dish in frame.dishes:
                for col in dish.colonies:
                    rows.append(
                        {
                            "timeseries": self.name,
                            "frame": frame_idx,
                            "timestamp": frame.timestamp,
                            "dish": dish.label,
                            "colony_uid": f"{self.name}_{dish.label}_{col.label}",
                            "x": col.centroid[0],
                            "y": col.centroid[1],
                            "radius": col.radius,
                            "state": col.state,
                            "expansion_rate": col.expansion_rate,
                            "age": col.age,
                        }
                    )

        return rows

    def delete(self):
        """
        _Unlinks stored images and clears temporary files. Run this after each timeseries if iterating over multiple days_
        """
        self.fg_masks = None
        self.bg_masks = None
        self.frames.clear()
        Image.clear_tmp_dir()

    @staticmethod
    def compute_features(
        stats: list | str | Path, save_path: str | Path = "", name: str = "timeseries"
    ) -> pd.DataFrame:
        """
        _Computes averaged features, like expansion rate, from stats_

        Args:
            stats (list | str | Path): _List of stats or path to where they are stored (as a csv file)_
            save_path (str | Path, optional): _Directory to where store the features csv_. Defaults to "".
            name (str, optional): _File name_. Defaults to "timeseries".

        Raises:
            ValueError: _description_
            TypeError: _description_

        Returns:
            pd.DataFrame: _Features dataframe_
        """

        # Loads in stats from python list or saved csv file
        valid_extensions = [".csv"]
        if isinstance(stats, list):
            df = pd.DataFrame(stats)
        elif isinstance(stats, (str, Path)):
            stats = Path(stats)
            if stats.is_file() and stats.suffix.lower() in valid_extensions:
                df = pd.read_csv(stats)
            else:
                raise ValueError("Invalid file path")
        else:
            raise TypeError("stats must be a dict or path to a .csv file")

        save_path = Path(save_path)

        # Converts timestamp to delta t
        df["t_rel"] = (
            df["timestamp"] - df.groupby("timeseries")["timestamp"].transform("min")
        ).dt.total_seconds() / 3600

        def _extract_colony_features(raw_df: pd.DataFrame) -> pd.Series:
            """
            _Helper for efficient feature extraction_
            """

            # removes temp colonies, keeps if ever perm/lost
            unique_states = set(raw_df["state"].unique())
            if unique_states == {"temp"}:
                return None

            t = raw_df["t_rel"].values
            r = raw_df["radius"].values
            g = raw_df["expansion_rate"].values

            delta_t = t[1] - t[0]
            g_norm = g / delta_t

            timeseries = raw_df["timeseries"].iloc[0]
            dish = raw_df["dish"].iloc[0]
            final_state = raw_df["state"].iloc[-1]

            appearance_time = t[0]

            max_growth = np.max(g_norm)
            mean_growth = np.mean(g_norm)
            growth_std = np.std(g_norm)

            final_radius = r[-1]
            max_radius = r.max()
            lifetime = t[-1] - t[0]

            return pd.Series(
                {
                    "timeseries": timeseries,
                    "dish": int(dish + 1),
                    "final_state": final_state,
                    "lag_time": appearance_time,
                    "max_expansion_rate": max_growth,
                    "mean_expansion_rate": mean_growth,
                    "expansion_std": growth_std,
                    "final_radius": final_radius,
                    "max_radius": max_radius,
                    "lifetime": lifetime,
                    "n_frames": len(raw_df),
                }
            )

        colony_df = (
            df.groupby("colony_uid", group_keys=False)
            .apply(_extract_colony_features)
            .reset_index()
        )

        colony_df.to_csv(os.path.join(save_path, f"{name}_features.csv"), index=False)

        return colony_df

    def export_mp4(
        self, save_path: str | Path = "", file_name: str = "tracking.mp4", fps: int = 30
    ) -> None:
        """
        _Exports mp4 file of timelapse with drawn colonies_

        Args:
            save_path (str | Path, optional): _Where to save the video_. Defaults to "".
            file_name (str, optional): _File name_. Defaults to "tracking.mp4".
            fps (int, optional): _How many frames per second the video should be in_. Defaults to 30.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        n_dishes = len(self.frames[0].dishes)

        def _process_frame(frame, dish_idx):
            """
            _Helper for converting color formats_
            """
            dish = frame.dishes[dish_idx]

            if dish.tracked_detection is None:
                return None

            img = dish.tracked_detection.load()

            # ensure 3 channels
            if len(img.shape) == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

            # BGR -> RGB
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # ensure even dimensions
            h, w = img.shape[:2]
            img = img[: h - h % 2, : w - w % 2]

            return img

        # creates mp4 per dish
        for d in range(n_dishes):
            output_path = save_path / f"dish_{d + 1}_{file_name}"

            writer = None

            try:
                for frame in tqdm(self.frames, desc=f"Dish {d + 1} MP4"):
                    img = _process_frame(frame, d)

                    if img is None:
                        continue

                    if writer is None:
                        writer = imageio.get_writer(
                            output_path, fps=fps, codec="libx264", quality=8
                        )

                    writer.append_data(img)

            finally:
                if writer is not None:
                    writer.close()
