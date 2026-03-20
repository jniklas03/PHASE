from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import KDTree
from itertools import repeat
from collections import defaultdict
import os
import pandas as pd

from .frame import Frame
from .colony import Colony, CostFunction
from .dish import Dish

from ..helpers.inputs import read_time, Image, create_circular_mask

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
        name:str, directory: str | Path,
        clip: int | float | None = None,
        clamp: int | float | None = None,
        max_images: int | None = None,
        sample_fraction: float | None = None
        ):
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
        timeseries.load_timeseries(directory, clip, clamp, max_images, sample_fraction)
        return timeseries

    def load_timeseries(self,
                        directory: str | Path,
                        clip: int | float | None = None,
                        clamp: int | float | None = None,
                        max_images: int | None = None,
                        sample_fraction: float | None = None):
        directory = Path(directory)
        if not directory.is_dir():
            raise TypeError("directory must be a string of directory path (str or Path).")

        if sample_fraction is not None and not (0 < sample_fraction <= 1):
            raise ValueError("fraction must be between 0 and 1.")
        
        if sum(x is not None for x in [clamp, max_images, sample_fraction]) > 1:
            raise ValueError("Only one of clip, max_images, or sample_fraction can be set.")

        valid_extensions = {".jpg", ".jpeg", ".png"}
        all_items = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions])

        if clip is not None:
            if isinstance(clip, int):
                if clip < 0:
                    raise ValueError("clip (int) must be >= 0.")
                all_items = all_items[clip:]  # remove first n images
            elif isinstance(clip, float):
                if not (0 <= clip <= 1):
                    raise ValueError("clip (float) must be between 0 and 1.")
                n = int(len(all_items) * clip)
                all_items = all_items[n:]  # remove first fraction
            else:
                raise TypeError("clip must be int or float.")

        if clamp is not None:
            if isinstance(clamp, int):
                if clamp <= 0:
                    raise ValueError("clip (int) must be > 0.")
                selected_items = all_items[:clamp]

            elif isinstance(clamp, float):
                if not (0 < clamp <= 1):
                    raise ValueError("clip (float) must be between 0 and 1.")
                n = max(1, int(len(all_items) * clamp))
                selected_items = all_items[:n]

        elif max_images is not None and max_images < len(all_items):
            # Use n_max_images for uniform sampling
            indices = np.linspace(0, len(all_items) - 1, max_images, dtype=int)
            selected_items = [all_items[i] for i in indices]
        elif sample_fraction is not None and sample_fraction < 1.0:
            # Use fraction for uniform sampling
            n_to_load = max(1, int(len(all_items) * sample_fraction))
            indices = np.linspace(0, len(all_items) - 1, n_to_load, dtype=int)
            selected_items = [all_items[i] for i in indices]
        else:
            # Load all images
            selected_items = all_items

        for item in tqdm(selected_items, desc="Loading frames"):
            timestamp = read_time(item.name)
            frame = Frame(
                name=item.stem,
                timestamp=timestamp,
                image=Image(item)
            )
            self.frames.append(frame)

    def generate_dishes_timeseries(self, use_stencil: bool = True):
        """
        populate dishes in each frame of the timeseries

        parameters
        ----------
        use_stencil : bool, optional
            if true, first frame acts as stencil for cropping (default True)
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
                self.frames = list(tqdm(
                    ex.map(lambda f: _populate_frame(f, stencils), self.frames[1:]),
                    total=len(self.frames[1:]),
                    desc="Generating dishes"
                ))

        else:
            # populate each frame independently
            with ThreadPoolExecutor() as ex:
                self.frames = list(tqdm(
                    ex.map(lambda f: _populate_frame_no_crop(), self.frames),
                    total=len(self.frames),
                    desc="Generating dishes"
                ))
    
    def make_masks(self, n=5):
        """
        generate foreground and background masks for dishes

        parameters
        ----------
        n : int, optional
            number of initial frames used to compute background masks (default 5).
        """

        # foreground mask from last frame
        def _make_fg_mask(dish):
            return dish.label, dish.isolate_fg()

        with ThreadPoolExecutor() as ex:
            results = list(tqdm(
                ex.map(_make_fg_mask, self.frames[-1].dishes),
                total=len(self.frames[-1].dishes),
                desc="Making foreground masks"
            ))

        fg_masks = [None] * len(results)
        for label, mask in results:
            fg_masks[label] = mask


        # bg mask from first n frames
        def _make_bg_mask(dish):
            return dish.label, dish.isolate_bg()

        frame_groups: dict[int, list[np.ndarray]] = defaultdict(list)

        dishes = [dish for frame in self.frames[:n] for dish in frame.dishes]

        with ThreadPoolExecutor() as ex:
            results = list(tqdm(
                ex.map(_make_bg_mask, dishes),
                total=len(dishes),
                desc="Making background masks"
            ))

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

    def preprocess_timeseries(self, use_bg_mask = True, use_fg_mask = True, use_area_filter = False, n=5):
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

        def _preprocess_frame(frame):
            for dish in frame.dishes:
                dish.preprocessed = Image(dish.preprocess_dish(
                    fg_mask=self.fg_masks[dish.label] if use_fg_mask else None,
                    bg_mask=self.bg_masks[dish.label] if use_bg_mask else None,
                    use_bg_mask=use_bg_mask,
                    use_fg_mask=use_fg_mask,
                    use_area_filter=use_area_filter
                ))
            return frame
        
        with ThreadPoolExecutor() as ex:
            list(tqdm(
                ex.map(_preprocess_frame, self.frames),
                total=len(self.frames),
                desc="Preprocessing frames"
            ))

    def detect_timeseries_old(self):
        for frame in tqdm(self.frames, desc="Detecting colonies"):
            for dish in frame.dishes:
                blobs = dish.detect_colonies()

                for blob in blobs:
                    colony = Colony(
                        centroid=(int(blob.pt[0]), int(blob.pt[1])),
                        radius=int(blob.size / 2),
                        expansion_rate=0,
                        label=self.get_new_label()
                    )
                    dish.colonies.append(colony)
                
                frame.count = len(dish.colonies)

    def detect_timeseries(
            self,
            distance_threshold: int = 10,
            detection_threshold: float = 0.5,
            min_lost_radius: int = 2,
            cost_function: CostFunction = CostFunction.IOU_CIRCLE,
            verbosity: int = 0
            ):
        
        # 0. grab dt for kalman filter (normalised for 30 mins)
        dt = (self.frames[1].timestamp - self.frames[0].timestamp).total_seconds() / 1800.0

        # 1. init first frame colonies
        for dish in self.frames[0].dishes:
            blobs = dish.detect_colonies()

            for blob in blobs:
                dish.colonies.append(Colony(
                    centroid=(int(blob.pt[0]), int(blob.pt[1])),
                    radius=float(blob.size / 2),
                    expansion_rate=0,
                    label=self.get_new_label(), # getting unique label from timeseries
                    state="temp",
                    age=1
                ))
        
        def _detect_worker(
                prev_dish,
                curr_dish,
                dt,
                distance_threshold = distance_threshold,
                detection_threshold = detection_threshold,
                min_lost_radius = min_lost_radius,
                cost_function = cost_function,
                verbosity = verbosity
                ) -> Dish:
            
            # 2. init prev and current colonies for dish pairs
            prev_cols = prev_dish.colonies
            curr_dish.colonies = []

            # 3. predict colony states
            predicted_cols = [c.predict(dt) for c in prev_cols]

            # 4. apply growth extrapolation to previously lost colonies, mask them, and detect colonies for current dish
            lost_mask = np.zeros_like(prev_dish.preprocessed.load())

            for col in predicted_cols:
                if col.state == "lost":

                    r = max(int(col.radius), 1) # failsafe: minimum radius of 1
                    x, y = int(col.centroid[0]), int(col.centroid[1])
                    
                    cv.circle(lost_mask, (x, y), r, 255, -1)

            preprocessed_masked = cv.bitwise_and(
                curr_dish.preprocessed.load(),
                cv.bitwise_not(lost_mask)
            )

            detected_blobs, _ = Dish.colony_detection(
                preprocessed_masked,
                curr_dish.crop.load()
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

                    neighbors = tree.query_ball_point(pred_col.centroid, distance_threshold)

                    for j in neighbors:

                        cost = cost_function(pred_col, detected_blobs[j])

                        if cost < detection_threshold:
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

            unmatched_pred = [predicted_cols[i] for i in range(n) if i not in matched_pred_idx] # colonies that disappeared
            unmatched_det = [detected_blobs[j] for j in range(m) if j not in matched_det_idx] # colonies that newly appeared

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
                if pred_col.radius >= min_lost_radius and (pred_col.state == "perm" or pred_col.state == "lost"):
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
                    missed_frames=0
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

                list(ex.map(
                    _detect_worker,
                    prev_frame.dishes,
                    curr_frame.dishes,
                    repeat(dt),
                    repeat(distance_threshold),
                    repeat(detection_threshold),
                    repeat(min_lost_radius),
                    repeat(cost_function),
                    repeat(verbosity)
                ))


    def export_images(self, save_path: str | Path = ""):
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
            cv.putText(overlay, str(dish.label), (x - 40, y - 40),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        cv.imwrite(str(save_path / "dish_detection" / f"{first_frame.name}_debug.png"), overlay)

        def _save_images(frame):
        # crops, preprocessed, detections
            for dish in frame.dishes:
                if dish.crop is not None:
                    cv.imwrite(str(save_path / "dish_detection" / f"{frame.name}_dish{dish.label}.png"), dish.crop.load())

                if dish.preprocessed is not None:
                    cv.imwrite(str(save_path / "preprocessed" / f"{frame.name}_dish{dish.label}.png"), dish.preprocessed.load())

                if dish.preprocessed_masked is not None:
                    cv.imwrite(str(save_path / "preprocessed_masked" / f"{frame.name}_dish{dish.label}.png"), dish.preprocessed_masked.load())

                if dish.initial_detection is not None:
                    cv.imwrite(str(save_path / "initial_detection" / f"{frame.name}_dish{dish.label}.png"), dish.initial_detection.load())

                if dish.tracked_detection is not None:
                    cv.imwrite(str(save_path / "tracked_detection" / f"{frame.name}_dish{dish.label}.png"), dish.tracked_detection.load())

        with ThreadPoolExecutor() as ex:
            list(tqdm(
                ex.map(_save_images, self.frames),
                total=len(self.frames),
                desc="Saving images"
            ))

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
                count = sum(
                    1 for col in (dish.colonies or [])
                    if col.state in {"perm", "lost"}
                )
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
    
    def export_stats(self):
        rows = []

        for frame_idx, frame in tqdm(enumerate(self.frames), desc="Exporting data"):
            for dish in frame.dishes:
                for col in dish.colonies:
                    rows.append({
                    "timeseries": self.name,
                    "frame": frame_idx,
                    "timestamp": frame.timestamp,
                    "dish": dish.label,
                    "colony_id": col.label,
                    "x": col.centroid[0],
                    "y": col.centroid[1],
                    "radius": col.radius,
                    "state": col.state,
                    "expansion_rate": col.expansion_rate,
                    "age": col.age
                })
        
        return rows

    def export_stats_parallel(self):

        def _export_worker(args):
            frame_idx, frame = args
            worker_rows = []
            for dish in frame.dishes:
                for col in dish.colonies:
                    worker_rows.append({
                    "timeseries": self.name,
                    "frame": frame_idx,
                    "timestamp": frame.timestamp,
                    "dish": dish.label,
                    "colony_id": col.label,
                    "x": col.centroid[0],
                    "y": col.centroid[1],
                    "radius": col.radius,
                    "state": col.state,
                    "expansion_rate": col.expansion_rate,
                    "age": col.age
                })
                    
            return worker_rows
    
        with ThreadPoolExecutor() as ex:
            worker_results = list(tqdm(
                ex.map(_export_worker, enumerate(self.frames)),
                total=len(self.frames),
                desc="Exporting stats"
            ))

        rows = [row for sublist in worker_results for row in sublist]

        return rows

    def delete(self):
        self.fg_masks = None
        self.bg_masks = None
        self.frames.clear()
        Image.clear_tmp_dir()

    def execute(
            self,
            name: str,
            directory: str | Path = "",
            use_stencil=True,
            use_bg_mask=True,
            use_fg_mask=True,
            use_area_filter=False,
            detection_threshold=0.5,
            distance_threshold=10,
            min_lost_radius=2,
            cost_function: CostFunction = CostFunction.IOU_CIRCLE,
            verbosity=0,
            save_path: str | Path = "",
            clip: int | float | None = None,
            clamp: int | float | None = None,
            max_images: int | None = None,
            sample_fraction: float | None = None,
            fps=60,

        ):
        directory = Path(directory)

        if not directory.is_dir():
            raise TypeError("directory must be a string of directory path (str or Path).")

        save_path.mkdir(parents=True, exist_ok=True)

        ts = Timeseries.from_directory(f"{name}", directory, clip, clamp, max_images, sample_fraction)

        ts.generate_dishes_timeseries(
            use_stencil=use_stencil)

        ts.preprocess_timeseries(
            use_bg_mask=use_bg_mask,
            use_fg_mask=use_fg_mask,
            use_area_filter=use_area_filter
            )

        ts.detect_timeseries(
            detection_threshold=detection_threshold,
            distance_threshold=distance_threshold,
            min_lost_radius=min_lost_radius,
            cost_function=cost_function,
            verbosity=verbosity
            )

        stats = ts.export_stats()
        pd.DataFrame(stats).to_csv(os.path.join(save_path, f"{name}_stats.csv"))

        ts.export_images(save_path=save_path)
        ts.plot_counts(save_path=save_path)
        ts.export_gif(fps=fps, save_path=save_path)

        ts.delete()
        del ts

        return stats

    def export_gif(
        self,
        save_path: str | Path = "",
        file_name: str = "tracking.gif",
        fps: int = 30
    ):
        """
        Generate one GIF per dish from tracked_detection images.

        Parameters
        ----------
        save_path : str | Path
            Output directory
        file_name : str
            Base name of the GIF file
        fps : int
            Frames per second
        """
        import imageio.v2 as imageio

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        duration = 1 / fps

        n_dishes = len(self.frames[0].dishes)

        # --- helper: load + fix image ---
        def _process_frame(frame, dish_idx):
            dish = frame.dishes[dish_idx]

            if dish.tracked_detection is None:
                return None

            img = dish.tracked_detection.load()

            # ensure 3 channels
            if len(img.shape) == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

            # FIX COLOR: BGR -> RGB
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            return img

        # --- per dish GIF creation ---
        for d in range(n_dishes):
            gif_frames = []

            with ThreadPoolExecutor() as ex:
                results = list(tqdm(
                    ex.map(lambda f: _process_frame(f, d), self.frames),
                    total=len(self.frames),
                    desc=f"Dish {d} GIF"
                ))

            gif_frames = [img for img in results if img is not None]

            if gif_frames:
                imageio.mimsave(
                    save_path / f"dish_{d}_{file_name}",
                    gif_frames,
                    duration=duration
                )