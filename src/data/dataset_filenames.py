import os
import pathlib
import itertools
from typing import Dict
from typing import List

import src.data.config


def generate_filenames(params: Dict[str, str]) -> List[str]:
    delim_prefix = "$"

    add_prefix = lambda param_values, param_name: [
        f"{param_name}{delim_prefix}{val}" for val in param_values
    ]

    combinations = list(
        itertools.product(*[add_prefix(params[param], param) for param in params])
    )

    result = []

    for combination in combinations:
        result.append(
            src.data.config.PARAM_SEP.join(
                param_val.split(delim_prefix)[-1] for param_val in combination
            )
            + "."
            + src.data.config.FILE_EXTENSION
        )

    return result


def extract_param_from_filename(filename: str, index: int) -> str:
    name_without_ext = filename.split(".", maxsplit=1)[0]
    return name_without_ext.split(src.data.config.PARAM_SEP)[index]


def extract_type_camera(filename: str) -> str:
    return extract_param_from_filename(
        filename, src.data.config.TYPE_CAMERA_PARAM_INDEX
    )


def extract_type_face(filename: str) -> str:
    return extract_param_from_filename(filename, src.data.config.TYPE_FACE_PARAM_INDEX)


def rename_files(
    path_to_dir: pathlib.Path, path_to_new_filenames: pathlib.Path
) -> None:
    with open(path_to_new_filenames, "r") as f:
        new_filenames = f.read().splitlines()

    old_filenames = [
        f
        for f in os.listdir(path_to_dir)
        if os.path.isfile(os.path.join(path_to_dir, f))
    ]
    old_filenames.sort()

    assert len(new_filenames) == len(old_filenames), "Array sizes don't match!"

    for f in old_filenames:
        os.rename(
            os.path.join(path_to_dir, f),
            os.path.join(path_to_dir, new_filenames.pop(0)),
        )
