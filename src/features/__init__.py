from typing import List
from typing import Union
from typing import Callable
from typing import Iterable
from collections.abc import Iterable

import numpy as np
import mahotas
import pywt
import skimage.feature

import src.data.utils


def calc_features(
    channels: Iterable[np.ndarray],
    feature_funcs: Iterable[Callable],
    blockwise: Union[None, int] = None,
) -> np.array:
    result = np.array([])

    for channel in channels:
        if blockwise is not None:
            blocks = get_blocks(channel, blockwise)
        else:
            blocks = [channel]

        for block in blocks:
            for feature_func in feature_funcs:
                feature_value = np.array(feature_func(block)).ravel()
                result = np.concatenate((result, feature_value), axis=None)

    return result.ravel()


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


def calc_and_get_haralick(arr: np.ndarray, axis: int = 0) -> np.array:
    return mahotas.features.haralick(arr).mean(axis)


def haralick_rdwt_features(img: np.ndarray, wavelet="haar") -> np.array:
    result_features = []

    cA, (cH, cV, cD) = pywt.dwt2(img, wavelet)
    img_cont = src.data.utils.resize_img(img.copy(), cA.shape)
    result_features.append(calc_and_get_haralick(img_cont))

    for a in [cA, cH, cV, cD]:
        result_features.append(calc_and_get_haralick(a.astype(np.uint8)))

    return np.asarray(result_features).ravel()


def haralick_rdwt_features_with_blockwise(
    img: np.ndarray, count_blocks: int = 9
) -> np.array:
    count_blocks = 9

    result_features = []
    blocks = get_blocks(img, count_blocks)
    img_cont = None

    for block in blocks:
        result_features.append(haralick_rdwt_features(block))

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


def lbp_hist_features(
    img: np.ndarray, radius: int = 10, n_points: int = 30, method: str = "uniform"
) -> np.array:
    lbp = skimage.feature.local_binary_pattern(img, n_points, radius, method)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    return hist
