from typing import List

import numpy as np
import mahotas
import pywt

import src.data.utils


def haralick_rdwt_features(img: np.ndarray) -> np.array:
    def calc_and_get_haralick(arr: np.ndarray) -> np.array:
        return mahotas.features.haralick(arr).mean(axis=0)

    def get_blocks(img: np.ndarray, count: int) -> List[np.ndarray]:
        M = img.shape[0] // int(np.sqrt(count))
        N = img.shape[1] // int(np.sqrt(count))
        blocks = []

        for item in [
            img[x : x + M, y : y + N]
            for x in range(0, img.shape[0], M)
            for y in range(0, img.shape[1], N)
        ]:
            blocks.append(item)

        return blocks

    count_blocks_for_split = 9

    result_features = []
    blocks = get_blocks(img, count_blocks_for_split)
    img_cont = None

    for block in blocks:
        cA, (cH, cV, cD) = pywt.dwt2(block, "haar")
        img_cont = src.data.utils.resize_img(block.copy(), cA.shape)
        result_features.append(calc_and_get_haralick(img_cont))

        for i, a in enumerate([cA, cH, cV, cD]):
            result_features.append(calc_and_get_haralick(a.astype(np.uint8)))

    return np.asarray(result_features).ravel()


def dispersion_feature(img) -> np.float64:
    return np.std(img)


def mean_feature(img) -> np.float64:
    return np.mean(img)


def median_feature(img) -> np.float64:
    return np.median(img)


def maximum_feature(img) -> np.float64:
    return np.max(img)


def minimum_feature(img) -> np.float64:
    return np.min(img)


def fraction_pixels_for_interval_feature(img, A=30, B=50) -> np.float64:
    flattened_img = img.copy().ravel()
    return sum((flattened_img > 30) & (flattened_img < 50)) / len(flattened_img)


def scatter_feature(img) -> np.float64:
    return np.max(img) - np.min(img)


def hist_feauture(img, bins=20, min_val=0, max_val=255) -> np.ndarray:
    return np.histogram(img, bins=bins, range=(min_val, max_val))[0]


def hist_gradient_feauture(img, bins=20, min_val=0, max_val=30) -> np.ndarray:
    return np.histogram(np.gradient(img.ravel()), bins=bins, range=(min_val, max_val))[
        0
    ]


def ratio_upper_lower_parts_feature(img) -> np.float64:
    half_index = len(img) // 2
    return np.mean(img[:half_index]) / (np.mean(img[half_index:]) + 1e-5)
