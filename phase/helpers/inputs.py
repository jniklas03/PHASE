import cv2 as cv, numpy as np, os
import re
from pypdf import PdfReader
from datetime import datetime
import sys
from pathlib import Path

def read_img(source):
    """
    Reads a string filepath or a numpy array.
    For ease of using image processing functions standalone or in pipelines.

    Returns numpy array / image.

    Parameters
    ----------
    source: str or np.ndarray
        Image, or string to the image path.

    Returns
    -------
    np.ndarray
        Image in numpy array form.
    """
    if (isinstance(source, (str, Path)) and os.path.isfile(source)):
        img = cv.imread(source)
    elif isinstance(source, np.ndarray):
        img = source
    else:
        raise TypeError("source must be a file path or a NumPy array")
    return img

def read_time(filename):
    """
    Extracts datetime from image filenames, i.e. '01.10.2025-17.40.02.jpg'
    For extracting timestamps from images.

    Returns datetime object, or None if pattern not found.

    Parameters
    ----------
    filename: str
        Filename of the file to extract the timestamp from, with the pattern DD.MM.YYYY-HH.MM.SS

    Returns
    -------
    datetime
        Datetime object with the timestamp.

    """
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)

    match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})-(\d{2})\.(\d{2})\.(\d{2})", name)
    if not match:
        return None

    day, month, year, hour, minute, second = map(int, match.groups())
    return datetime(year, month, day, hour, minute, second)

def read_image_paths(directory):
    """
    Extracts all image files (.jpg, .jpeg, .png) from a given directory.
    For ease of use in pipelines that depend on multiple image files.

    Returns full image paths as well as base names.

    Parameters
    ----------
    directory: str
        String of a directory.

    Returns
    -------
    list
        List of full image paths.
    list
        List of base names of images.
    """
    if not isinstance(directory, str) or not os.path.isdir(directory):
        raise TypeError("directory must be a string of directory path.")
    
    valid_extensions = {".jpg", ".jpeg", ".png"}
    
    image_paths = []
    base_names = []

    for img in sorted(os.listdir(directory)): # sorts all entries from given directory

        full_path = os.path.join(directory, img) # grabs full path

        # checks if entry is a file and contains a valid extension
        if os.path.isfile(full_path) and os.path.splitext(img.lower())[1] in valid_extensions:

            image_paths.append(full_path)
            base_names.append(os.path.splitext(img)[0])

    return image_paths, base_names

def read_pdf(
        source,
        save_images = False,
        save_path = "",
        file_name = "from_pdf",
        regex_name_pattern = r"(WT\s*\d+\s+P\s*\d+)"
):
    """
    Reads pdf data exported by the Interscience cell counter and extracts images, dish names, and cell counts.
    The data can be then used as ground truth for validation.

    Returns a dictionary of the dish names with the associated value counts.

    Parameters
    ----------
    source: str
        String of a filepath to the pdf file.
    save_images: bool, default = False
        Whether to save the detected images.
    save_path: str, default = ""
        String of a path to where the detected images should be saved.
    file_name: str, default = "from_pdf"
        String of the file name the images should be saved as.
    regex_name_pattern: str, default = r"(WT\\s*\\d+\\s+P\\s*\\d+)"
        Regular expression pattern of the dish naming scheme.

    Returns
    -------
    dict
        Dictionary of plate names with associated colony counts.

    """
    if save_images:
        save_folder = save_path + r"\FromPDF"
        os.makedirs(save_folder, exist_ok=True)

    pattern_count = re.compile(r"Count :\s*(\d+)")
    pattern_name = re.compile(regex_name_pattern)

    doc = PdfReader(source)

    all_counts = []
    all_names = []

    for page in doc.pages:
        text = page.extract_text()

        count = int(pattern_count.findall(text)[0])
        all_counts.append(count)

        name = str(pattern_name.findall(text)[0]).replace(" ", "_")
        all_names.append(name)

        if save_images:
            for idx, image in enumerate(page.images[1:2], start=1): #change the slice, and naming to get all images
                save_name = f"{file_name}_{idx+1}"
                with open(f"{os.path.join(save_folder, save_name)}.jpg", "wb") as fp:
                    fp.write(image.data)

    output = dict(zip(all_names, all_counts))

    return(output)

def show_image(
        source,
        name = "image"
):
    cv.imshow(f"{name}", source)
    key = cv.waitKey(0)
    if key == ord("e"):
        sys.exit("Exiting")
    elif key == ord("s"):
        cv.imwrite(f"{name}.png", source)