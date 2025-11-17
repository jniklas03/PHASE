import cv2 as cv
import numpy as np
import os
import warnings

from ..helpers.inputs import read_img, show_image

def preprocess(
        source,
        mask = None,
        fg_mask = None,
        bg_mask = None,
        area_filter = True,
        s = 121,
        C = 11,
        kernel_size = 3,
        min_area = 5,
        max_area = 200,
        save = False,
        save_path = "",
        file_name = "preprocessed",
        idx: int = None
        ):
    """
    Preprocesses input image of cropped dish into a thresholded binary image of colonies.

    Returns preprocessed image.
    
    Parameters
    ----------
    source: str or np.ndarray
        Image of dish, or string to the image path.
    mask: np.ndarray, optional
        Mask of background area outside of dish, if None the background crop won't be applied and watershedding won't work.
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
    idx: int, optional
        Passed by a wrapper function when processing mutliple dishes.    
    
    Returns
    -------
    np.ndarray
        Preprocessed dish.
    """
    img = read_img(source=source)

    if save and not save_path:
        warnings.warn(f"No specified save path. Images saved in the current directory ({os.getcwd()}) under ...Preprocessing.")

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
    if area_filter:
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(threshold, connectivity=8)
        filtered = np.zeros_like(threshold)

        for i in range(1, num_labels):  # skips background
            area = stats[i, cv.CC_STAT_AREA]
            if min_area <= area <= max_area:
                filtered[labels == i] = 255
    else:
        filtered = threshold
        # watershed here?

    # "crops" the outside of the dish if mask is passed
    if mask is not None:
        cropped = cv.bitwise_and(filtered, filtered, mask=mask)
    else:
        cropped = filtered

    # applies fg_mask and bg_mask if passed
    if fg_mask is not None:
            masked1 = cv.bitwise_and(cropped, cropped, mask=fg_mask)
    if bg_mask is not None:
            masked2 = cv.bitwise_and(masked1, masked1, mask=cv.bitwise_not(bg_mask))
    else:
        masked2 = cropped
    
    # erosion
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv.morphologyEx(masked2, cv.MORPH_ERODE, kernel)

    # saving
    save_name = f"{file_name}_preprocessed_{idx}.png" if idx is not None else f"{file_name}_preprocessed.png"

    if save:
        save_path_preprocessing = os.path.join(save_path, "Preprocessing")
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, save_name), eroded)

    print(f"File {save_name} preprocessed.")

    return eroded

def preprocess_fg_isolation(
        source,
        mask = None,
        kernel_size = 500,
        save = False,
        save_path = "",
        file_name = "preprocessed",
        idx: int = None
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
    idx: int, optional
        Passed by a wrapper function when processing mutliple dishes.  

    Returns
    -------
    np.ndarray
        Preprocessed dish.
    """
    img = read_img(source=source)

    if save and not save_path:
        warnings.warn(f"No specified save path. Images saved in the current directory ({os.getcwd()}) under ...Preprocessing.")

    green_channel = img[:, :, 1]

    blur = cv.medianBlur(green_channel, 5)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    tophat = cv.morphologyEx(blur, cv.MORPH_TOPHAT, kernel)

    _, threshold = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    if mask is not None:
        threshold = cv.bitwise_and(threshold, threshold, mask=mask)

    save_name = f"{file_name}_p{idx}.png" if idx is not None else f"{file_name}_p.png"

    if save:
        save_path_preprocessing = os.path.join(save_path, "Preprocessing")
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, save_name), threshold)
    print(f"File {save_name} preprocessed.")

    return threshold

def preprocess_bg_isolation(
        source,
        mask,
        s = 121,
        C = 11,
        kernel_size = 3,
        min_area = 5,
        max_area = 200,
        save=False,
        save_path = "",
        file_name = "preprocessed",
        idx: int = None
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
    idx: int, optional
        Passed by a wrapper function when processing mutliple dishes.  

    Returns
    -------
    np.ndarray
        Preprocessed dish.
    """
    img = read_img(source=source)

    if save and not save_path:
        warnings.warn(f"No specified save path. Images saved in the current directory ({os.getcwd()}) under ...Preprocessing.")

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

    if mask is not None:
        filtered = cv.bitwise_and(opened, opened, mask=mask)
    else:
        filtered = opened

    save_name = f"{file_name}_p{idx}.png" if idx is not None else f"{file_name}_p.png"

    if save:
        save_path_preprocessing = os.path.join(save_path, "Preprocessing")
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, save_name), filtered)
    print(f"File {save_name} preprocessed.")

    return filtered