from typing import Tuple

import numpy as np
import cv2


def get_coords_of_face_area(img: np.ndarray) -> Tuple[int]:
    HAARCASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

    # faces = faceCascade.detectMultiScale(
    #     img,
    #     scaleFactor=1.05,
    #     minNeighbors=20,
    #     minSize=(55, 55),
    #     flags=cv2.CASCADE_SCALE_IMAGE,
    # )

    # faces = faceCascade.detectMultiScale(
    #     img,
    #     scaleFactor=1.1,
    #     minNeighbors=3,
    #     minSize=(30, 30),
    #     flags=cv2.CASCADE_SCALE_IMAGE,
    # )

    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    assert len(faces) > 0

    x, y, w, h = faces[0]

    return x, y, w, h


def extract_box(img: np.ndarray, coords: Tuple[int]) -> np.ndarray:
    x, y, w, h = coords
    return img[y : y + h, x : x + w]


def get_face_area(img: np.ndarray, coords: Tuple[int]) -> np.ndarray:
    face_img = extract_box(img, coords)
    return face_img
