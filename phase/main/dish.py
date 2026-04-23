from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
import warnings

from pathlib import Path

from ..helpers.inputs import read_img, Image
from .colony import Colony


@dataclass
class Dish:
    centroid: tuple[int, int]
    radius: int
    label: int | None = None
    count: int | None = None
    colonies: list[Colony] = field(default_factory=list)
    crop: Image | None = None
    preprocessed: Image | None = None
    preprocessed_masked: Image | None = None
    initial_detection: Image | None = None
    tracked_detection: Image | None = None

    def _mask_from_crop(self) -> np.ndarray:
        h, w = self.crop.load().shape[:2]
        cx, cy = w // 2, h // 2

        mask = np.zeros((h, w), dtype=np.uint8)
        cv.circle(mask, (cx, cy), self.radius, 255, -1)

        return mask

    def isolate_fg(self) -> np.ndarray:
        """
        Generates a foreground mask

        Returns:
            np.ndarray: Foreground mask
        """
        mask = self._mask_from_crop()

        preprocessed = Dish.fg_isolation(self.crop.load())

        cropped = cv.bitwise_and(preprocessed, preprocessed, mask=mask)

        return cropped

    def isolate_bg(self) -> np.ndarray:
        """
        Generates a background mask

        Returns:
            np.ndarray: Background mask
        """
        mask = self._mask_from_crop()

        preprocessed = Dish.bg_isolation(self.crop.load())

        cropped = cv.bitwise_and(preprocessed, preprocessed, mask=mask)

        return cropped

    def preprocess_dish(
        self,
        fg_mask: np.ndarray,
        bg_mask: np.ndarray,
        use_bg_mask: bool = True,
        use_fg_mask: bool = False,
        kernel_size: int = 3,
        use_area_filter: bool = True,
    ) -> np.ndarray:
        """
        Final preprocessing method for a dish.

        Args:
            fg_mask (np.ndarray): _description_
            bg_mask (np.ndarray): _description_
            use_bg_mask (bool, optional): _description_. Defaults to True.
            use_fg_mask (bool, optional): _description_. Defaults to False.
            kernel_size (int, optional): _description_. Defaults to 3.
            use_area_filter (bool, optional): _description_. Defaults to True.

        Returns:
            np.ndarray: _Preprocessed dish_
        """
        # flipping the outside of the dish
        mask = self._mask_from_crop()

        preprocessed = Dish.preprocessing(
            self.crop.load(), use_area_filter=use_area_filter
        )

        cropped = cv.bitwise_and(preprocessed, preprocessed, mask=mask)

        result = cropped

        if use_bg_mask or use_fg_mask:
            kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )

        if use_fg_mask:
            result = cv.bitwise_and(result, result, mask=fg_mask)

        if use_bg_mask:
            result = cv.bitwise_and(result, result, mask=cv.bitwise_not(bg_mask))

        if use_bg_mask or use_fg_mask:
            result = cv.morphologyEx(result, cv.MORPH_ERODE, kernel)

        return result

    def detect_colonies(self) -> int:
        """
        Dish wrapper for colony detection.

        Returns:
            int: Number of colonies
        """
        assert self.preprocessed is not None, (
            "Preprocessed image not found. Run preprocessing first."
        )

        if self.crop is None:
            warnings.warn("Crop not found. Using preprocessed image for visualisation.")
            blobs, output = Dish.colony_detection(self.preprocessed.load())
        else:
            blobs, output = Dish.colony_detection(
                self.preprocessed.load(), raw_img=self.crop.load()
            )

        self.initial_detection = Image(output)

        return blobs

    def draw_tracked_colonies(self, verbosity: int = 0):
        """
        Draws detections on raw dish crop based on verbosity

        Args:
            verbosity (int, optional): 0 just draws the circles around the detected colonies. 1 adds the ID. 2 addtionally displays furter statistics. Defaults to 0.
        """
        if self.crop is None:
            return

        output = self.crop.load().copy()

        for col in self.colonies:
            kp = cv.KeyPoint(
                float(col.centroid[0]),
                float(col.centroid[1]),
                float(max(col.radius, 1) * 2),
            )

            # Color based on state
            if col.state == "temp":
                color = (255, 0, 0)  # Blue
            elif col.state == "perm":
                color = (0, 255, 0)  # Green
            else:  # "lost"/"merged"
                color = (0, 0, 255)  # Red

            cv.drawKeypoints(
                output,
                [kp],
                output,
                color=color,
                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            # Draw text based on verbosity
            text = ""
            if verbosity == 1:
                text = f"ID:{col.label}"
            elif verbosity == 2:
                text = f"ID:{col.label} | R:{int(col.radius)} | A:{col.age} | S:{col.state}"

            if text:
                cx, cy = col.centroid
                cv.putText(
                    output,
                    text,
                    (cx + int(col.radius) + 2, cy),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (0, 0, 0),
                    1,
                    cv.LINE_AA,
                )

        self.tracked_detection = Image(output)

    @staticmethod
    def preprocessing(
        source: np.ndarray | str | Path,
        s: int = 121,
        C: int = 11,
        use_area_filter: bool = False,
        min_area: int = 5,
        max_area: int = 200,
    ) -> np.ndarray:
        """
        Base, general purpose preprocessing function.

        Args:
            source (np.ndarray | str | Path): Image or path to image to apply processing to
            s (int, optional): Blocksize for adaptive thresholding. Defaults to 121.
            C (int, optional): Substractive constant for thresholding. Defaults to 11.
            use_area_filter (bool, optional): Keep only colonies between min and max_area. Deprecated, don't use. Defaults to False.
            min_area (int, optional): Minimum colony area for area filter. Defaults to 5.
            max_area (int, optional): Maximum colony area for area filter. Defaults to 200.

        Returns:
            np.ndarray: Preprocessed dish
        """
        img = read_img(source=source)

        # isolating green channel
        green_channel = img[:, :, 1]

        # thresholding
        threshold = cv.adaptiveThreshold(
            src=green_channel,
            maxValue=255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv.THRESH_BINARY_INV,
            blockSize=s,
            C=C,
        )

        # area filter if area_filter flag is passed, otherwise watershed
        if use_area_filter:
            num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
                threshold, connectivity=8
            )
            filtered = np.zeros_like(threshold)

            for i in range(1, num_labels):  # skips background
                area = stats[i, cv.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    filtered[labels == i] = 255
        else:
            filtered = threshold
            # watershed not implemented

        return filtered

    @staticmethod
    def fg_isolation(
        source: np.ndarray | str | Path, kernel_size: int = 500
    ) -> np.ndarray:
        """
        Base processing function for foreground isolation.

        Args:
            source (np.ndarray | str | Path): Source of image for processing
            kernel_size (int, optional): Kernel size for tophat. Defaults to 500.

        Returns:
            np.ndarray: Processed foreground isolated image
        """
        img = read_img(source=source)

        green_channel = img[:, :, 1]

        blur = cv.medianBlur(green_channel, 5)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

        tophat = cv.morphologyEx(blur, cv.MORPH_TOPHAT, kernel)

        _, threshold = cv.threshold(
            tophat, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

        kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

        opened = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel_open)

        return opened

    @staticmethod
    def bg_isolation(
        source: np.ndarray | Path | str,
        s=121,
        C=11,
        kernel_size=3,
        min_area=5,
        max_area=200,
    ) -> np.ndarray:
        """
        Base processing function for background isolation.

        Args:
            source (np.ndarray | Path | str): Source of image for processing
            s (int, optional): Blocksize for adaptive thresholding. Defaults to 121.
            C (int, optional): Substractive constant for thresholding. Defaults to 11.
            kernel_size (int, optional): Kernel size for morphological opening. Defaults to 3.
            min_area (int, optional): Keeps everything below that value. Defaults to 5.
            max_area (int, optional): Keeps everything above that value. Defaults to 200.

        Returns:
            np.ndarray: Processed image
        """
        img = read_img(source=source)

        green_channel = img[:, :, 1]

        threshold = cv.adaptiveThreshold(
            src=green_channel,
            maxValue=255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv.THRESH_BINARY_INV,
            blockSize=s,
            C=C,
        )

        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            threshold, connectivity=8
        )
        filtered = np.zeros_like(threshold)

        for i in range(1, num_labels):  # skip background
            area = stats[i, cv.CC_STAT_AREA]
            if not min_area <= area <= max_area:
                filtered[labels == i] = 255

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv.morphologyEx(filtered, cv.MORPH_OPEN, kernel)

        return opened

    @staticmethod
    def colony_detection(
        source: np.ndarray | str | Path, raw_img=None
    ) -> tuple[int, np.ndarray]:
        """
        Base function for detecting colonies using blob detection

        Args:
            source (np.ndarray | str | Path): Image or source of image to detect colonies
            raw_img (np.ndarray, optional): Raw image to draw colony detections on. If none, source will be used for visualisation. Defaults to None.

        Returns:
            tuple[int, np.ndarray]: Colony counts, image with drawn detections
        """
        img = read_img(source=source)

        if raw_img is None:
            raw_img = img

        params = cv.SimpleBlobDetector_Params()  # Values from hyperparameter tuning

        params.minThreshold = 0
        params.maxThreshold = 255  # Smaller values = less false positives
        params.thresholdStep = 1  # Smaller values = more true positives

        params.filterByArea = True  # Area in pxs
        params.minArea = 1
        params.maxArea = 750

        params.filterByColor = True
        params.blobColor = 255  # Detects white colonies

        params.filterByCircularity = (
            True  # how much does the geometrical shape fit the form of a circle
        )
        params.minCircularity = 0.1

        params.filterByConvexity = (
            True  # "fullness" of the circle; think of a pie chart
        )
        params.minConvexity = 0.7

        params.filterByInertia = (
            True  # how elongated is the circle - lower values mean more elongated.
        )
        params.minInertiaRatio = 0.1

        detector = cv.SimpleBlobDetector_create(params)  # Creates detector object
        blobs = detector.detect(img)  # Blobs are markers around colonies

        output = cv.drawKeypoints(
            raw_img,
            blobs,
            np.array([]),
            (0, 255, 0),
            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )  # Output = initial image with colonies marked

        return blobs, output
