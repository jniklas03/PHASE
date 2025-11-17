import cv2 as cv
import numpy as np
import os
import warnings

from ..helpers.inputs import read_img

def sort_circles(circles, row_tolerance=100):
    """
    Sorts detected dishes from top left to bottom right.

    Parameters:
    circles: list of np.ndarray
        Detected dishes.
    row_tolerance: int
        Value for how strictly dishes should be grouped together in a row.

    Returns
    -------
    list of np.ndarray
        Sorted detected dishes.
    """
    circles = sorted(circles, key=lambda c: c[1])

    rows = []
    current_row = [circles[0]]

    for c in circles[1:]:
        if abs(c[1] - current_row[-1][1]) < row_tolerance:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]
    rows.append(current_row)

    for row in rows:
        row.sort(key=lambda c: c[0])
    sorted_circles = [c for row in rows for c in row]
    return sorted_circles

def crop(image, coordinates):
    """
    Crops around dishes in an image.
    
    Parameters
    ----------
    image: np.ndarray
        Image to be cropped.
    coordinates: tuple
        Coordinates of the detected circles.

    Returns
    -------
    list
        Cropped dishes.
    list
        Masks of background.

    """
    dishes = [] # list for the cropped images
    masks = [] # list for masks

    img = read_img(image)

    for (x, y, r) in coordinates:
        mask = np.zeros(img.shape[:2], dtype=np.uint8) # mask (black image) the size of the input image
        cv.circle(mask, (x, y), r, 255, -1) # fills the mask with a white circle at the location of the coordinates (detected dish)

        masked_img = cv.bitwise_and(  # applies mask (keeps values where the mask is white) to original image
            img, 
            img, 
            mask=mask)

        x1, y1 = max(0, x-r), max(0, y-r) # defines top left corner of the crop
        x2, y2 = min(img.shape[1], x+r), min(img.shape[0], y+r) # defines bottom right corner of the crop

        square_crop = masked_img[y1:y2, x1:x2] # applies a square crop for the masked dishes
        mask_crop = mask[y1:y2, x1:x2] # crops the masks

        dishes.append(square_crop)
        masks.append(mask_crop)
        
    return dishes, masks

def detect_dishes(
        source,
        save=True,
        save_path = "",
        file_name = "dish_detected", 
        metadata: dict = None,
        debug = False
):
    """
    Detects dishes in an image and crops around them.
    
    Parameters
    ----------
    source: str or np.ndarray
        Raw image, or string of the image path.
    save: bool, default=True
        Whether to save the cropped dishes.
    save_path: str, optional
        Path to directory where the images and metadata are saved.
    file_name: str, optional
        Name to save the dishes as.
    metadata: dict, optional
        Metadata dictionary handled by main.py.
    debug: bool, default=False
        Whether to save the input image with numbered circles for debugging.

    Returns
    -------
    list of np.ndarray
        List of cropped dishes.
    list of np.ndarray
        List of masks.
    list of tuples
        List of coordinates of the crops.
    dict
        Metadata.
    """
    if (save and not save_path) or (debug and not save_path):
        warnings.warn(f"No specified save path. Images saved in the current directory ({os.getcwd()}) under ...Dishes.")

    dishes, masks, coordinates = [], [], []

    img = read_img(source=source)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    clahe_obj = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahed = clahe_obj.apply(gray_img)

    save_path_dish_detection = os.path.join(save_path, "Dishes") # path for dish crops

    blur = cv.medianBlur(clahed, 21) # blur so that hough circles doesn't detect random stuff

    circles = cv.HoughCircles( # creates a numpy array of detected circles
        blur, # image, should be grayscale
        cv.HOUGH_GRADIENT, # detection method
        dp=3, # resolution used for the detection; dp=2 means half resolution of the original image
        minDist=800, # minimum distance between the centers of circles in px
        param1=125, # upper threshold for canny edge detection (uses canny edge detection internally)
        param2=100, # threshold for center detection, turn this up if non-dishes are detected
        minRadius=400, # minimum and maxmimum radius in px
        maxRadius=600 
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int") 
        circles = sort_circles(circles, row_tolerance=150)

        coordinates = [(x, y, r) for (x, y, r) in circles]
        dishes, masks = crop(img, coordinates)

        if debug:
            debug_img = img.copy()
            for idx, (x,y,r) in enumerate(coordinates, start=1):
                cv.circle(debug_img, (x, y), r, (0, 255, 0), 4)  # draws outline
                cv.circle(debug_img, (x, y), 10, (0, 0, 255), -1)  # draws center point
                cv.putText( # draws number label
                    debug_img, str(idx), (x - 40, y - 40), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        for idx, (dish, mask) in enumerate(zip(dishes, masks), start=1): # circles defined by their x and y coordinates of their center as well as their radius; idx is used for naming the files
            if metadata:
                x, y, r = coordinates[idx-1]
                metadata[file_name][idx] = [{
                    "center": [int(x), int(y)],
                    "radius": int(r),
                    "colony_count": None,
                }]


            save_name = f"{file_name}_dish_{idx}.png" if idx is not None else f"{file_name}_dish.png"

            if save: # saving the dishes if the flag is passed
                os.makedirs(save_path_dish_detection, exist_ok=True)
                cv.imwrite(os.path.join(save_path_dish_detection, save_name), dish)

        print(f"{len(circles)} dishes detected in file: {file_name}.")

        if debug: # saves debug image
            os.makedirs(save_path_dish_detection, exist_ok=True)
            cv.imwrite(os.path.join(save_path_dish_detection, f"{file_name}_debug.png"), debug_img)

    else:
        warnings.warn("No dishes detected.")

    return dishes, masks, coordinates, metadata

