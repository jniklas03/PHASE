import cv2 as cv
import numpy as np
import os

from ..image_manipulation.dish_detection import detect_dishes, crop
from ..image_manipulation.preprocessing import preprocess_bg_isolation, preprocess_fg_isolation

def make_masks(
        image_paths,
        save_path = "",
        save = False,
        n_to_stack=5
):
    """
    Makes foreground and background masks from a given image
    """
    # FOREGROUND MASKING
    
    last_image_path = image_paths[-1]

    dishes, masks, coordinates, _ = detect_dishes( # dish crops from first image
        source=last_image_path,
        file_name=os.path.basename(last_image_path),
        save=False,
        save_path=save_path,
        debug=False
    )

    file_name = os.path.splitext(os.path.basename(last_image_path))[0]
    foreground_masks = [preprocess_fg_isolation(source=dish, mask=mask, file_name=file_name, kernel_size=500) for dish, mask in zip(dishes, masks)]

    if save:
        for idx, mask in enumerate(foreground_masks):
            os.makedirs(os.path.join(save_path, "Masks"), exist_ok=True)
            cv.imwrite(os.path.join(save_path, "Masks", f"fg_mask{idx+1}.png"), mask)

    # BACKGROUND MASKING
    first_n_image_paths = image_paths[:n_to_stack] # grabs the first n images

    first_n_dishes = []
    
    for img_path in first_n_image_paths: # crops dishes and preprocesses them
        dishes, masks = crop(img_path, coordinates)

        preprocessed = [preprocess_bg_isolation(source=dish, mask=mask, file_name=os.path.splitext(os.path.basename(img_path))[0]) for dish, mask in zip(dishes, masks)]
        first_n_dishes.append(preprocessed)

    background_masks = []

    for i in range(len(first_n_dishes[0])): # combines all preprocessed images for a given dish (OR operand); results in masks of background/noise
        group = [d[i] for d in first_n_dishes]
        
        stack = np.zeros_like(group[0])
        for idx, img in enumerate(group):
            stack = cv.bitwise_or(stack, img)
            if save:
                os.makedirs(os.path.join(save_path, "Masks"), exist_ok=True)
                cv.imwrite(os.path.join(save_path, "Masks", f"bg_mask_dish{i+1}_{idx+1}.png"), img)
        background_masks.append(stack)
        if save:
            os.makedirs(os.path.join(save_path, "Masks"), exist_ok=True)
            cv.imwrite(os.path.join(save_path, "Masks", f"bg_mask_dish{i+1}_stack.png"), stack)

    return foreground_masks, background_masks, coordinates

class DishState:
    def __init__(self, fine_buffer = 2):
        self.fine = True
        self.history = []
        self.fine_buffer = fine_buffer
    def trigger(self):
        self.fine_buffer -= 1
        if self.fine_buffer <= 0:
            self.fine = False

def check_state(dish_states, window=3, threshold=10):
    for state in dish_states:
        if len(state.history) < window:
            continue
        
        _, counts = zip(*state.history[-window:])

        if sum(count > threshold for count in counts) >= 2:
            if counts[-1] > counts[-2]:
                state.trigger()