from typing import List, Callable

import numpy as np


def apply_feature_funcs(x: np.ndarray, feature_funcs: List[Callable]) -> np.array:
    result = np.array([])

    for feature_func in feature_funcs:
        result = np.append(result, feature_func(x))

    return result
