from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
import os
import warnings

from ..helpers.inputs import read_img

from .dish import Dish

@dataclass
class Frame:
    name: str
    timestamp: datetime
    image_path: Path | str
    image: np.ndarray | None = None
    dishes: list[Dish] = field(default_factory=list)

    def load_image(self): # lazy loading
        if self.image is None:
            self.image = read_img(self.image_path)
    
    def generate_dishes(
            self,
            save: bool = False,
            save_path: Path | str = "",
            debug: bool = False
    ):
        self.load_image()

        self.dishes = dish_generation(
            source=self.image,
            save=save,
            save_path=save_path,
            file_name=self.name,
            debug=debug
        )
    
    def generate_dishes_from_crop(self, stencils):
        self.load_image()

        crops = dish_cropping(self.image, stencils)

        for stencil, crop in zip(stencils, crops):
            self.dishes = [
                Dish(
                    label=stencil.label,
                    centroid=stencil.centroid,
                    radius=stencil.radius,
                    crop=crop
                )
            ]

def dish_detection(img: np.ndarray) -> list[Dish]:
    """
    Detects circular dishes in an image using Hough Circle Transform.

    Parameters
    ----------
    source: str or np.ndarray
        Raw image, or string of the image path.

    Returns
    -------
    list of Dish
        List of Dish objects with centroid and radius filled.
    """
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    clahe_obj = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahed = clahe_obj.apply(gray_img)

    blur = cv.medianBlur(clahed, 21) # blur so that hough circles doesn't detect random stuff

    centroids = cv.HoughCircles( # creates a numpy array of detected circles
        blur, # image, should be grayscale
        cv.HOUGH_GRADIENT, # detection method
        dp=3, # resolution used for the detection; dp=2 means half resolution of the original image
        minDist=800, # minimum distance between the centers of circles in px
        param1=125, # upper threshold for canny edge detection (uses canny edge detection internally)
        param2=100, # threshold for center detection, turn this up if non-dishes are detected
        minRadius=400, # minimum and maxmimum radius in px
        maxRadius=600 
    )

    dishes = []

    if centroids is not None:

        centroids = np.round(centroids[0, :]).astype(int)

        for x,y,r in centroids:
            dishes.append(Dish(
                label=None,
                centroid=(x,y),
                radius=r)
                )

    else:
        warnings.warn("No dishes detected.")

    return dishes

def dish_sorting(dishes: list[Dish], row_tolerance=100) -> list[Dish]:
    """
    Sorts detected dishes from top left to bottom right.

    Parameters:
    dishes: list of class(Dish)
    row_tolerance: int
        Value for how strictly dishes should be grouped together in a row (in pixels).

    Returns
    -------
    list of class(Dish)
        Sorted dishes.
    """
    if not dishes:
        warnings.warn("Dishes list empty!")
        return []

    dishes = sorted(dishes, key=lambda d: d.centroid[1])

    rows = []
    current_row = [dishes[0]]

    for dish in dishes[1:]:
        if abs(dish.centroid[1] - current_row[-1].centroid[1]) < row_tolerance:
            current_row.append(dish)
        else:
            rows.append(current_row)
            current_row = [dish]
    rows.append(current_row)

    sorted_dishes = []

    for row in rows:
        row.sort(key=lambda d: d.centroid[0])
        sorted_dishes.extend(row)

    for idx, dish in enumerate(sorted_dishes):
        dish.label = idx

    return sorted_dishes

def dish_cropping(source, stencils: list[Dish]) -> list[Dish]:
    """
    Crops around dishes in an image.

    Parameters
    ----------
    source: str or np.ndarray
        Image path or array to be cropped.
    dishes: list of Dish
        List of Dish objects containing centroid and radius.

    Returns
    -------
    list of Dish
        List of Dish objects with the crop attribute filled.
    """

    img = read_img(source)
    crops = []

    for stencil in stencils:
        x, y = stencil.centroid
        r = stencil.radius

        # mask (black image) the size of the input image
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # fills the mask with a white circle at the location of the coordinates (detected dish)
        cv.circle(mask, (x, y), r, 255, -1)

        # applies mask (keeps values where the mask is white) to original image
        masked_img = cv.bitwise_and(img, img, mask=mask)

        # defines top left corner of the crop
        x1, y1 = max(0, x-r), max(0, y-r)

        # defines bottom right corner of the crop
        x2, y2 = min(img.shape[1], x+r), min(img.shape[0], y+r)

        crop = masked_img[y1:y2, x1:x2]
        crops.append(crop)

    return crops

def dish_generation(
        source,
        save=True,
        save_path = "",
        file_name = "dish_detected", 
        debug = False
):
    """
    Detects dishes in an image, sorts them correctly, and crops around them.
    
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
    debug: bool, default=False
        Whether to save the input image with numbered circles for debugging.

    Returns
    -------
    list of Dish
        List of Dish objects with crop and label attributes filled.
    """
    if (save or debug) and not os.path.isdir(save_path):
        save_path = ""
        warnings.warn(f"No specified save path. Images saved in the current directory ({os.getcwd()}) under ...Dishes.")

    save_path = os.path.join(save_path, "Dishes") # path for dish crops

    img = read_img(source)

    dishes = dish_detection(img)

    print(f"{len(dishes)} dishes detected in file: {file_name}.")

    if not dishes:
        return []

    dishes = dish_sorting(dishes)

    dishes = dish_cropping(img, dishes)

    if debug:
        debug_img = img.copy()

        for dish in dishes:
            x,y = dish.centroid
            r = dish.radius

            cv.circle(debug_img, (x, y), r, (0, 255, 0), 4)  # draws outline
            cv.circle(debug_img, (x, y), 10, (0, 0, 255), -1)  # draws center point

            cv.putText( # draws number label
                debug_img,
                str(dish.label),
                (x - 40, y - 40),
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                3,
            )

        os.makedirs(save_path, exist_ok=True)

        cv.imwrite(
            os.path.join(save_path, f"{file_name}_debug.png"),
            debug_img,
        )

    if save:
        os.makedirs(save_path, exist_ok=True)

        for dish in dishes:
            save_name = f"{file_name}_dish_{dish.label}.png"
            cv.imwrite(os.path.join(save_path, save_name), dish.crop)

    return dishes