import os
import pathlib
from typing import List
from typing import Tuple

import numpy as np
import cv2


def is_non_zero_file(filepath: pathlib.Path) -> bool:
    return os.path.isfile(filepath) and os.path.getsize(filepath) > 0


def get_directory_file_list(dirpath: pathlib.Path) -> List[str]:
    return [
        f
        for f in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, f)) and not f.startswith(".")
    ]


def resize_img(
    img: np.ndarray, size: Tuple[int, int], interpolation: int = cv2.INTER_AREA
) -> np.ndarray:
    return cv2.resize(img, dsize=size, interpolation=interpolation)
