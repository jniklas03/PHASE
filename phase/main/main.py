import cv2 as cv
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

from ..helpers.timelapse import make_masks, DishState, check_state
from ..helpers.inputs import read_time, read_image_paths
from ..helpers.plotting import init_plot, update_live_plot

from ..image_manipulation.preprocessing import preprocess, preprocess_fg_isolation
from ..image_manipulation.dish_detection import detect_dishes, crop

from ..colony_detection.counting import detect_colonies

def pipeline(
        source,
        save_path = "",
        save_metadata=False,
        save_dishes=False,
        save_preprocessed=False,
        save_detected=True
        ):
    """
    Process image to get yield cropped dishes, with circled colonies.

    Parameters
    ----------
    source: str
        Filepath to the image file of the petri dishes with the colonies.
    kernel_size: int, optional
        Kernel size for opening; higher number yields more smoothed, generally better results, but takes longer.
    save_path: str, optional
        Filepath where the images should be saved. Creates different folders for dish crops, preprocessed images, and dishes with detected colonies.
    save_metadata: bool, default=False
        Whether to save metadata - dish crop positions and number of colonies.
    save_dishes: bool, default=False
        Whether to save the dish crops.
    save_preprocessed: bool, default=False
        Whether to save the preprocessed images.
    save_detected: bool, default=True
        Whether to save the image with the detected colonies.

    """
    if not os.path.isfile(source):
        raise TypeError("source needs to be a string of a filepath.")

    img = cv.imread(source)
    file_name = os.path.splitext(os.path.basename(source))[0]

    metadata = {file_name: {}}

    dishes, masks, coordinates, dish_metadata = detect_dishes(
        source=img,
        save=save_dishes,
        save_path=save_path,
        file_name=file_name,
        metadata=metadata
    )
    
    preprocessed = []

    for idx, dish in enumerate(dishes):
        preprocessed.append(preprocess(
            source=dish,
            mask=masks[idx],
            area_filter=False,
            save=save_preprocessed,
            save_path=save_path,
            file_name=file_name,
            idx=idx+1
        ))

    for idx, preprocessed_img in enumerate(preprocessed):
        colony_metadata = detect_colonies(
            source=preprocessed_img,
            raw_img=dishes[idx],
            save=save_detected,
            save_path=save_path,
            file_name=file_name,
            idx=idx+1
        )

    if save_metadata:
        metadata_file = os.path.join(save_path, "metadata.yaml")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = yaml.safe_load(f) or {}
        else:
            metadata = {}

        metadata.update(dish_metadata)

        with open(metadata_file, "w") as f:
            yaml.safe_dump(metadata, f)

def mult_pipeline(
        source,
        n_dishes = 6,
        kernel_size=500, 
        save_path = "",
        save_metadata=False,
        save_dishes=False,
        save_preprocessed=False,
        save_detected=True
                  ):
    for file in os.listdir(source):
        pipeline(
            source=os.path.join(source, file),
            n_dishes = n_dishes,
            kernel_size = kernel_size, 
            save_path = save_path,
            save_metadata = save_metadata,
            save_dishes = save_dishes,
            save_preprocessed=save_preprocessed,
            save_detected=save_detected
        )

def timelapse_pipeline(
        source,
        save_intermediates = False,
        save_path = "",
        n_to_stack = 5,
        plot = False,
        fine_buffer = 3,
        use_masks = True
):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    image_paths, file_names = read_image_paths(source)

    if use_masks:
        fg_masks, bg_masks, coordinates = make_masks(
            image_paths=image_paths,
            save_path=save_path,
            save=save_intermediates,
            n_to_stack=n_to_stack
        )
    else:
        _, _, coordinates, _ = detect_dishes(
            source=image_paths[-1],
            save=save_intermediates,
            save_path=save_path
        )

    if plot:
        fig, ax = init_plot()

    n_dishes = len(coordinates)

    dish_states = [DishState(fine_buffer) for _ in range(n_dishes)]

    t0 = None

    for img_path, file_name in zip(image_paths, file_names):

        # handling time steps

        t = read_time(img_path)

        if t0 is None:
            t0 = t
        else:
            pass

        delta_t = (t - t0).total_seconds() / 3600.0

        # cropping and preprocessing

        dishes, masks = crop(img_path, coordinates)

        # going through each dish and applying preprocessing and masks, dependant on growth state

        for idx, (dish, mask) in enumerate(zip(dishes, masks)):
            if save_intermediates:
                os.makedirs(os.path.join(save_path, "Dishes"), exist_ok=True)
                cv.imwrite(os.path.join(save_path, "Dishes", f"{file_name}_dish_{idx+1}.jpg"), dish)

            preprocessed = preprocess(source=dish,
                        mask=mask,
                        fg_mask=fg_masks[idx] if use_masks else None,
                        bg_mask=bg_masks[idx] if use_masks else None,
                        area_filter=dish_states[idx].fine,
                        file_name=file_name,
                        save=save_intermediates,
                        save_path=save_path,
                        idx=idx+1
                        )
            if not dish_states[idx].fine:
                preprocessed = cv.morphologyEx(preprocessed, cv.MORPH_ERODE, kernel)

            count, _ = detect_colonies(source=preprocessed, raw_img=dish, save=save_intermediates, save_path=save_path, file_name=file_name, idx=idx+1)
            dish_states[idx].history.append((delta_t, count))        

        if plot:
            dish_counts_plot = {i: [(timestamp, count) for timestamp, count in dish_states[i].history] for i in range(n_dishes)}
            update_live_plot(dish_counts_plot, fig, ax)
            
        check_state(dish_states)
    if plot:
        plt.ioff()
        plt.show()
    
    return(None)
