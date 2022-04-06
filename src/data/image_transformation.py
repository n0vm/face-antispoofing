import os
import pathlib
import operator
from typing import List

import cv2
import numpy as np
import Levenshtein

import src.data.config
import src.data.utils


def read_points_for_transform(filepath: pathlib.Path):
    with open(filepath) as file:
        points_raw_list = file.read().splitlines()

    points_raw_list.remove("")
    points = [list(map(int, p.split(","))) for p in points_raw_list]
    half_index = len(points) // 2
    points_src, points_dst = points[:half_index], points[half_index:]

    return np.float32(points_src), np.float32(points_dst)


def perform_projective_transformation(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    transformation_matrix: np.ndarray,
):
    result = cv2.warpPerspective(
        image_src, transformation_matrix, (image_dst.shape[1], image_dst.shape[0])
    )

    return result


def make_image_transformation(
    filepath_src: pathlib.Path,
    filepath_dst: pathlib.Path,
    transformation_matrix: np.ndarray,
    homography_mode=True,
):
    image_src = cv2.imread(str(filepath_src))
    image_dst = cv2.imread(str(filepath_dst))
    return perform_projective_transformation(
        image_src, image_dst, transformation_matrix
    )


def get_transformation_matrix(exp_id: str, matrices_dir_path: pathlib.Path):
    exp_matrix_file = exp_id + "_matrix.npy"

    if not os.path.exists(str(matrices_dir_path / exp_matrix_file)):
        result_matrix_file = exp_matrix_file
    else:
        result_matrix_file = choose_transformation_matrix(exp_id, matrices_dir_path)

    with open(
        matrices_dir_path / result_matrix_file,
        "rb",
    ) as f:
        result = np.load(f)

    return result, result_matrix_file


def choose_transformation_matrix(exp_id: str, matrices_dir_path: pathlib.Path):
    matrix_files = src.data.utils.get_directory_file_list(matrices_dir_path)
    similarity_of_experiments = {}
    curr_exp_id_cont = None

    for matrix_file in matrix_files:

        # if not matrix_file.startswith("exp21"):
        #     continue

        curr_exp_id_cont = matrix_file.replace("_matrix.npy", "")

        # оцениваем схожесть параметров экспериментов
        similarity_of_experiments[matrix_file] = Levenshtein.distance(
            exp_id, curr_exp_id_cont
        )

    sorted_similarity_of_experiments = dict(
        sorted(similarity_of_experiments.items(), key=operator.itemgetter(1))
    )

    result_matrix_file = list(sorted_similarity_of_experiments.keys())[0]

    return result_matrix_file
