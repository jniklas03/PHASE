import cv2 as cv
import numpy as np
import os
import warnings

from ..helpers.inputs import read_img

def detect_colonies(
        source,
        raw_img = None,
        save = True,
        save_path = "",
        file_name = "colonies_detected",
        idx: int = None
):
    """
    Detects colonies.

    Returns number of colonies, image with detected colonies and metadata.

    Parameters
    ----------
    source: str or np.ndarray
        Thresholded image or string to the image path.
    raw_img: np.ndarray, optional
        Initial image, used for saving. If None, image from source will be used for visualisation.
    save: bool, default=False
        Whether to save the image with detected colonies.
    save_path: str, optional
        Path to directory where the image and metadata are saved.
    file_name: str, optional
        File name for the saved image.
    metadata: dict, optional
        Metadata dictionary handled by a wrapper function.
    idx: int, optional
        Passed by a wrapper function when processing mutliple dishes. 

    Returns
    -------
    int
        Number of detected colonies
    np.ndarray
        Image with detected colonies
    """
    img = read_img(source=source)

    if raw_img is None:
        raw_img = img

    if save and not save_path:
        warnings.warn(f"No specified save path. Images saved in the current directory ({os.getcwd()}) under ...Colonies.")

    params = cv.SimpleBlobDetector_Params() # Values from hyperparameter tuning

    params.minThreshold = 0
    params.maxThreshold = 255 # Smaller values = less false positives
    params.thresholdStep = 1 # Smaller values = more true positives

    params.filterByArea = True # Area in pxs
    params.minArea = 1
    params.maxArea = 750

    params.filterByColor = True
    params.blobColor = 255 # Detects white colonies

    params.filterByCircularity = True # how much does the geometrical shape fit the form of a circle
    params.minCircularity = 0.1

    params.filterByConvexity = True # "fullness" of the circle; think of a pie chart
    params.minConvexity = 0.7

    params.filterByInertia = True # how elongated is the circle - lower values mean more elongated.
    params.minInertiaRatio = 0.1

    detector = cv.SimpleBlobDetector_create(params) # Creates detector object
    blobs = detector.detect(img) # Blobs are markers around colonies

    output = cv.drawKeypoints(raw_img, blobs, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Output = initial image with colonies marked

    save_name = f"{file_name}_colonies_{idx}.png" if idx is not None else f"{file_name}_colonies.png"

    if save: # Saving images with marked colonies
        save_path_blob_detection = os.path.join(save_path, "Colonies")
        os.makedirs(save_path_blob_detection, exist_ok=True)
        cv.imwrite(os.path.join(save_path_blob_detection, save_name), output)

    print(f"{len(blobs)} colonies detected in file {save_name}.")

    return len(blobs), output