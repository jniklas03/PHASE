from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
import os
import warnings

from ..helpers.inputs import read_img, Image

from .dish import Dish

@dataclass
class Frame:
    name: str
    timestamp: datetime
    image: Image | None = None
    dishes: list[Dish] = field(default_factory=list)

    def populate_frame(self):
        self.dishes = dish_pipeline(source=self.image.load())

    def populate_frame_from_crop(self, stencils):
        self.dishes = crop_dishes(self.image.load(), stencils)

def detect_dishes(img: np.ndarray) -> list[Dish]:
    """

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

def sort_dishes(dishes: list[Dish], row_tolerance=100) -> list[Dish]:
    """

    """
    if not dishes:
        warnings.warn("Dish list empty!")
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

def crop_dishes(source: np.ndarray, stencils: list[Dish]) -> list[Dish]:
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

    dishes = []

    for idx, stencil in enumerate(stencils):
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

        dishes.append(Dish(
            label=idx,
            centroid=(x,y),
            radius=r,
            crop=Image(crop))
            )
        
    return dishes

def dish_pipeline(
        source
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
    img = read_img(source)

    dishes = detect_dishes(img)


    if not dishes:
        return []

    dishes = sort_dishes(dishes)

    dishes = crop_dishes(img, dishes)

    return dishes