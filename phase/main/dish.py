from dataclasses import dataclass

import cv2 as cv
import numpy as np
import os
import warnings

from ..helpers.inputs import read_img

@dataclass
class Dish:
    label: int | None
    centroid: tuple[int, int]
    radius: int
    count: int = 0
    colonies: list | None = None
    crop: np.ndarray | None = None
    preprocessed: np.ndarray | None = None

    def _mask_from_crop(self) -> np.ndarray:
        h, w = self.crop.shape[:2]
        cx, cy = w // 2, h // 2

        mask = np.zeros((h, w), dtype=np.uint8)
        cv.circle(mask, (cx, cy), self.radius, 255, -1)

        return mask
    
    def isolate_fg(self):
        mask = self._mask_from_crop()

        preprocessed = fg_isolation(self.crop)

        cropped = cv.bitwise_and(preprocessed, preprocessed, mask=mask)

        return cropped

    def isolate_bg(self):
        mask = self._mask_from_crop()

        preprocessed = bg_isolation(self.crop)

        cropped = cv.bitwise_and(preprocessed, preprocessed, mask=mask)

        return cropped

    def preprocess(self, fg_mask, bg_mask, use_bg_sep = True, kernel_size = 3, use_area_filter = True):
        # flipping the outside of the dish
        mask = self._mask_from_crop()

        preprocessed = preprocessing(self.crop, use_area_filter=use_area_filter)

        cropped = cv.bitwise_and(preprocessed, preprocessed, mask=mask)

        if use_bg_sep:
            fg_masked = cv.bitwise_and(cropped, cropped, mask=fg_mask)
            bg_masked = cv.bitwise_and(fg_masked, fg_masked, mask=cv.bitwise_not(bg_mask))
            
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
            eroded = cv.morphologyEx(bg_masked, cv.MORPH_ERODE, kernel)

            return eroded

        return cropped

def preprocessing(
        source,
        s = 121,
        C = 11,
        use_area_filter = True,
        min_area = 5,
        max_area = 200
        ):
    """
    Preprocesses input image of cropped dish into a thresholded binary image of colonies.

    Returns preprocessed image.
    
    Parameters
    ----------
    source: str or np.ndarray
        Image of dish, or string to the image path.
    fg_mask: np.ndarray, optional
        "Ground truth positive" mask of colonies from a time point in the future. From preprocess_fg_isolation()
    bg_mask: np.ndarray, optional
        "Ground truth negative" mask of the petri dish, marker, and other artifacts. From preprocess_bg_isolation().
    area_filter: bool, default=True,
        Filter large objects by virtue of connected components. Useful for just formed colonies to reduce the noise; turn off when colonies are larger.
    s: int, default=121
        Block size for thresholding. Bigger numbers include more to threshold.
    C: int, default=11
        Constant to subtract from thresholding.
    kernel_size: int, optional
        Kernel size for erosion. Used for noise removal and colony separation.
    save: bool, default=True
        Whether to save the preprocessed image.
    save_path: str, optional
        Path where the image should be saved.
    file_name: str, optional
        Name to save the preprocessed image as. 
    
    Returns
    -------
    np.ndarray
        Preprocessed dish.
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
        C=C
    )

    # area filter if area_filter flag is passed, otherwise watershed
    if use_area_filter:
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(threshold, connectivity=8)
        filtered = np.zeros_like(threshold)

        for i in range(1, num_labels):  # skips background
            area = stats[i, cv.CC_STAT_AREA]
            if min_area <= area <= max_area:
                filtered[labels == i] = 255
    else:
        filtered = threshold
        # watershed here?

    return filtered

def fg_isolation(
        source,
        kernel_size = 500
        ):
    """
    Preprocesses input image of cropped dish into a thresholded binary image. 
    Used on images of large colonies to be used as the positive ground truth.

    Returns preprocessed image.
    
    Parameters
    ----------
    source: str or np.ndarray
        Image of dish, or string of to the image path.
    mask: np.ndarray, optional
        Mask of background area outside of dish, if None the background crop won't be applied and watershedding won't work.
    kernel_size: int, default = 200
        Kernel size for tophat; higher number results in a smoother background and contrasted colonies, but takes longer. 
    save: bool, default = True
        Whether to save the preprocessed image.
    save_path: str, optional
        Path to directory where the image is saved.
    file_name: str, optional
        Name to save the preprocessed image as.

    Returns
    -------
    np.ndarray
        Preprocessed dish.
    """
    img = read_img(source=source)

    green_channel = img[:, :, 1]

    blur = cv.medianBlur(green_channel, 5)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    tophat = cv.morphologyEx(blur, cv.MORPH_TOPHAT, kernel)

    _, threshold = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    return threshold

def bg_isolation(
        source,
        s = 121,
        C = 11,
        kernel_size = 3,
        min_area = 5,
        max_area = 200
        ):
    """
    Preprocesses input image of cropped dish into a thresholded binary image. 
    Used on images with no colonies to be used as the negative ground truth.

    Returns preprocessed image.
    
    Parameters
    ----------
    source: str or np.ndarray
        Image of dish, or string of to the image path
    mask: np.ndarray, optional
        Mask of background area outside of dish, if None the background crop won't be applied and watershedding won't work.
    s: int, default=121
        Block size for thresholding. Bigger numbers include more to threshold.
    C: int, defualt=11
        Constant to subtract from thresholding.
    kernel_size: int, optional
        Kernel size for opening. Used for closing gaps.
    save: bool, default=True
        Whether to save the preprocessed images.
    save_path: str, optional
        Path to directory where the image is saved.
    file_name: str, optional
        Name to save the preprocessed image as.

    Returns
    -------
    np.ndarray
        Preprocessed dish.
    """
    img = read_img(source=source)

    green_channel = img[:, :, 1]

    threshold = cv.adaptiveThreshold(
        src=green_channel,
        maxValue=255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY_INV,
        blockSize=s,
        C=C
    )

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(threshold, connectivity=8)
    filtered = np.zeros_like(threshold)

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv.CC_STAT_AREA]
        if not min_area <= area <= max_area:
            filtered[labels == i] = 255

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv.morphologyEx(filtered, cv.MORPH_OPEN, kernel)

    return opened