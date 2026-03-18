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
    def from_directory(cls: type["Timeseries"], name:str, directory: str | Path, max_images: int | None = None, sample_fraction: float | None = None):
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
        timeseries.load_timeseries(directory, max_images, sample_fraction)
        return timeseries

    def load_timeseries(self, directory: str | Path, max_images: int | None = None, sample_fraction: float | None = None):
        directory = Path(directory)
        if not directory.is_dir():
            raise TypeError("directory must be a string of directory path (str or Path).")

        if sample_fraction is not None and not (0 < sample_fraction <= 1):
            raise ValueError("fraction must be between 0 and 1.")

        valid_extensions = {".jpg", ".jpeg", ".png"}
        all_items = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions])

        if max_images is not None and max_images < len(all_items):
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
            detection_threshold = 0.99,
            distance_threshold = 10,
            verbosity = 0,
            cost_function: CostFunction = CostFunction.IOU_CIRCLE,
            min_lost_radius = 2
            ):

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

        # worker function for parallelisation
        def _track_dish(prev_dish, curr_dish, detection_threshold, distance_threshold, cost_function, min_lost_radius, verbosity):

            # previous colonies get loaded in
            prev_cols = prev_dish.colonies

            # current colonies get initialised / cleared
            curr_dish.colonies = []

            # 2. apply growth extrapolation to previously lost colonies, mask them, and detect colonies for current dish
            lost_mask = np.zeros_like(prev_dish.preprocessed.load())

            for col in prev_cols:
                if col.state == "lost":

                    col.kalman_predict()

                    r = max(int(col.kf_radius), 1) # failsafe: minimum radius of 1
                    x, y = int(col.kf_centroid[0]), int(col.kf_centroid[1])
                    
                    cv.circle(lost_mask, (x, y), r, 255, -1)

            preprocessed_masked = cv.bitwise_and(
                curr_dish.preprocessed.load(),
                cv.bitwise_not(lost_mask)
            )

            curr_blobs, _ = Dish.colony_detection(
                preprocessed_masked,
                curr_dish.crop.load()
            )

            curr_dish.preprocessed_masked = Image(preprocessed_masked)

            # 3. make candidate pairs using KDTree (prev_blobs x curr_blobs) (replacement of hungarian)
            n, m = len(prev_cols), len(curr_blobs)
            matches = []

            if n > 0 and m > 0:

                curr_centroids = np.array([b.pt for b in curr_blobs])

                tree = KDTree(curr_centroids)

                candidate_pairs = []

                # distance gating
                for i, prev_col in enumerate(prev_cols):

                    neighbors = tree.query_ball_point(prev_col.centroid, distance_threshold)

                    for j in neighbors:

                        cost = cost_function(prev_col, curr_blobs[j])

                        if cost < detection_threshold:
                            candidate_pairs.append((cost, i, j))

                # sorting (lowest cost first)
                candidate_pairs.sort()

                used_prev = set()
                used_curr = set()

                for cost, i, j in candidate_pairs:

                    if i not in used_prev and j not in used_curr:
                        matches.append((i, j))
                        used_prev.add(i)
                        used_curr.add(j)

            # 4. assignment
            matched_prev = [prev_cols[r] for r, _ in matches]
            matched_curr = [curr_blobs[c] for _, c in matches]

            matched_prev_idx = {r for r, _ in matches}
            matched_curr_idx = {c for _, c in matches}

            unmatched_prev = [prev_cols[i] for i in range(n) if i not in matched_prev_idx] # colonies that disappeared
            unmatched_curr = [curr_blobs[j] for j in range(m) if j not in matched_curr_idx] # colonies that newly appeared

            # 5. link matched colonies
            for prev_col, curr_blob in zip(matched_prev, matched_curr):
                measured_radius = float(curr_blob.size / 2)
                measured_centroid = curr_blob.pt

                # update Kalman filter with new measurement
                prev_col.kalman_update(measured_radius, measured_centroid)

                curr_dish.colonies.append(Colony(
                    centroid=(int(prev_col.kf_centroid[0]), int(prev_col.kf_centroid[1])),
                    radius=prev_col.kf_radius,
                    label=prev_col.label,
                    state="perm",
                    age=prev_col.age + 1,
                    expansion_rate=prev_col.kf_expansion_rate,
                    P=prev_col.P,
                    Q=prev_col.Q,
                    R=prev_col.R
                ))

            # 6. newly lost colony handling
            for col in unmatched_prev:
                if col.radius >= min_lost_radius:
                    max_growth_per_frame = 1.5
                    col.kf_radius += min(col.kf_expansion_rate, max_growth_per_frame)

                    col.kf_expansion_rate *= 0.65

                    curr_dish.colonies.append(Colony(
                        centroid=(int(col.kf_centroid[0]), int(col.kf_centroid[1])),
                        radius=col.kf_radius,
                        label=col.label,
                        state="lost",
                        age=col.age,
                        expansion_rate=col.kf_expansion_rate,
                        P=col.P,
                        Q=col.Q,
                        R=col.R
                    ))

            # 7. add new colonies
            for blob in unmatched_curr:
                new_radius = float(blob.size / 2)
                new_centroid = blob.pt

                colony = Colony(
                    centroid=(int(new_centroid[0]), int(new_centroid[1])),
                    radius=new_radius,
                    label=self.get_new_label(),
                    state="temp",
                    age=1,
                    expansion_rate=0.0
                )

                colony.kf_centroid = np.array(new_centroid, dtype=float)
                colony.kf_radius = new_radius
                colony.kf_expansion_rate = 0.0

                curr_dish.colonies.append(colony)


            # 8. update dish count and draw tracked colonies
            curr_dish.count = len(curr_dish.colonies)

            curr_dish.draw_tracked_colonies(verbosity=verbosity)

            return curr_dish

        # iteratation over [1:] frames with worker
        for n in tqdm(range(1, len(self.frames)), desc="Tracking colonies"):

            prev_frame = self.frames[n - 1]
            curr_frame = self.frames[n]

            with ThreadPoolExecutor() as ex:

                list(ex.map(
                    lambda args: _track_dish(*args),
                    [
                        (prev_dish, curr_dish, detection_threshold, distance_threshold, cost_function, min_lost_radius, verbosity)
                        for prev_dish, curr_dish in zip(prev_frame.dishes, curr_frame.dishes)
                    ]
                ))

    def detect_timeseries_new(self,
                distance_threshold: int = 10,
                detection_threshold: float = 0.5,
                min_lost_radius: int = 2,
                cost_function: CostFunction = CostFunction.IOU_CIRCLE,
                verbosity: int = 0
                ):
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
            predicted_cols = [c.predict() for c in prev_cols]

            # 4. segment current image
            labels = curr_dish.segment(predicted_cols, get_new_label=self.get_new_label())

            # 5. convert segments into colonies
            detected_blobs = []
            centroids, radii = Colony.convert_segments_to_colonies(labels)

            for (x, y), r in zip(centroids, radii):
                detected_blobs.append(cv.KeyPoint(
                    x = float(x),
                    y = float(y),
                    size = float(r*2)
                ))

            # 6. building trees and assigning candidates
            # 6.1 make candidate pairs using KDTree (prev_blobs x curr_blobs) (replacement of hungarian)
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

            # 6.2. assignment
            matched_pred = [predicted_cols[r] for r, _ in matches]
            matched_det = [detected_blobs[c] for _, c in matches]

            matched_pred_idx = {r for r, _ in matches}
            matched_det_idx = {c for _, c in matches}

            unmatched_pred = [predicted_cols[i] for i in range(n) if i not in matched_pred_idx] # colonies that disappeared
            unmatched_det = [detected_blobs[j] for j in range(m) if j not in matched_det_idx] # colonies that newly appeared

            # 7. handling 3 possible states and updating kalman
            # 7.1. link matched colonies
            for pred_col, det_blob in zip(matched_pred, matched_det):
                measured_radius = float(det_blob.size / 2)
                measured_centroid = det_blob.pt

                # update Kalman filter with new measurement
                pred_col.update(measured_centroid, measured_radius)

                if pred_col.age >= 3:
                    pred_col.state = "perm"

                curr_dish.colonies.append(pred_col)


            # 7.2. newly lost colony handling
            for pred_col in unmatched_pred:
                if pred_col.radius >= min_lost_radius and pred_col.state == "perm":
                    pred_col.state = "lost"
                    curr_dish.colonies.append(pred_col)

            # 7.3. add new colonies
            for blob in unmatched_det:
                new_radius = float(blob.size / 2)
                new_centroid = blob.pt

                colony = Colony(
                    centroid=(int(new_centroid[0]), int(new_centroid[1])),
                    radius=new_radius,
                    label=self.get_new_label(),
                    state="temp",
                    age=1,
                    expansion_rate=0.0
                )

                curr_dish.colonies.append(colony)

            # 8. update dish count and draw tracked colonies
            curr_dish.count = len(curr_dish.colonies)

            curr_dish.draw_tracked_colonies(verbosity=verbosity)

            return curr_dish

        # 9. iteratation over [1:] frames with worker
        with ThreadPoolExecutor() as ex:
            for n in tqdm(range(1, len(self.frames)), desc="Tracking colonies"):

                prev_frame = self.frames[n - 1]
                curr_frame = self.frames[n]

                list(ex.map(
                    _detect_worker,
                    prev_frame.dishes,
                    curr_frame.dishes,
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