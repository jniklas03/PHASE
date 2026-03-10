from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor

from scipy.signal import savgol_filter

from .frame import Frame
from .colony import Colony, CostFunction
from .dish import Dish

from ..helpers.inputs import read_time, Image

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
            threshold = 0.95,
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

        # 2. iterate over subsequent frames
        for n in tqdm(range(1, len(self.frames)), desc="Tracking colonies"):
            prev_frame = self.frames[n - 1]
            curr_frame = self.frames[n]

            for prev_dish, curr_dish in zip(prev_frame.dishes, curr_frame.dishes):

                # previous colonies get loaded in
                prev_cols = prev_dish.colonies
                # current colonies get initialised / cleared
                curr_dish.colonies = []

        # 3. apply growth extrapolation to previously lost colonies, mask them, and detect colonies for current dish
                lost_mask = np.zeros_like(prev_dish.preprocessed.load())

                for col in prev_cols:
                    if col.state == "lost":
                        col.kalman_predict()
                        r = max(int(col.kf_radius), 1) # failsafe: minimium radius of 1
                        x, y = int(col.centroid[0]), int(col.centroid[1])

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


        # 4. make cost matrix (prev_blobs x curr_blobs) / run hungarian algorithm
                n, m = len(prev_cols), len(curr_blobs)
                matches = [] # clear matches for each dish

                if n > 0 and m > 0:
                    cost_matrix = np.zeros((n,m))

                    # calculating cost between each blob
                    for i, prev_colony in enumerate(prev_cols):
                        for j, curr_blob in enumerate(curr_blobs):
                            cost_matrix[i, j] = cost_function(prev_colony, curr_blob)

                    row_idx, col_idx = linear_sum_assignment(cost_matrix)

                    # thresholding
                    for row, col in zip(row_idx, col_idx):
                        if cost_matrix[row, col] < threshold:
                            matches.append((row, col))

                # assignment
                matched_prev = [prev_cols[r] for r, _ in matches]
                matched_curr = [curr_blobs[c] for _, c in matches]

                unmatched_prev = [prev_cols[i] for i in range(n) if i not in {r for r, _ in matches}] # colonies that disappeared
                unmatched_curr = [curr_blobs[j] for j in range(m) if j not in {c for _, c in matches}] # colonies that newly appeared

        # 5. link matched colonies
                for prev_col, curr_blob in zip(matched_prev, matched_curr):
                    curr_radius = float(curr_blob.size / 2)

                    # expansion_rate calculation
                    prev_col.kalman_update(curr_radius)
                    expansion_rate = prev_col.kf_radius - prev_col.radius

                    curr_dish.colonies.append(Colony(
                        centroid=(int(curr_blob.pt[0]), int(curr_blob.pt[1])),
                        radius=float(curr_blob.size / 2),
                        label=prev_col.label, # links labels
                        state="perm", # updates state to permanent
                        age=prev_col.age + 1, # increments age
                        expansion_rate=expansion_rate,
                        P=prev_col.P,
                        Q=prev_col.Q,
                        R=prev_col.R,
                    ))
        
        # 6. newly lost colony handling
                for col in unmatched_prev:
                    # area filtering to prevent noise from persisting
                    if col.radius >= min_lost_radius:
                        col.kalman_predict()
                        predicted_radius = col.kf_radius

                        curr_dish.colonies.append(Colony(
                            centroid=col.centroid,
                            radius=predicted_radius,
                            label=col.label,
                            state="lost", # changes state to lost
                            age=col.age, # keeps age static
                            expansion_rate=col.expansion_rate,
                            P=col.P,
                            Q=col.Q,
                            R=col.R
                        ))
        # 7. add new colonies
                for i, blob in enumerate(unmatched_curr):
                    curr_dish.colonies.append(Colony(
                        centroid=(int(blob.pt[0]), int(blob.pt[1])),
                        radius=float(blob.size / 2),
                        label=self.get_new_label(),
                        state="temp",
                        age=1
                    ))
        # 8. update dish count and draw tracked colonies
                curr_dish.count = len(curr_dish.colonies)
                curr_dish.draw_tracked_colonies(verbosity=verbosity)

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