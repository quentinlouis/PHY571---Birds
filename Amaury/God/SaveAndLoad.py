import json
import os
import pathlib
from typing import List, Union

import numpy as np

from God.Bird import Bird
from God.Pandora import NumpyEncoder
from God.Sky import Sky

def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def make_path_available(path: str) -> None:
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)


def save_data_dirname(data: Union[list, dict], output_dir: str, output_file: str) -> None:
    return save_data(data, os.path.join(output_dir, output_file))


def save_data(data: Union[list, dict], output_file: str) -> None:
    make_path_available(output_file)
    with open(output_file, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)


def load_data_dirname(input_dir: str, input_file: str) -> Union[list, dict]:
    return load_data(os.path.join(input_dir, input_file))


def load_data(input_file: str) -> Union[list, dict]:
    f = open(input_file, "r")
    content = ""
    for piece in read_in_chunks(f):
        content += piece
    return json.loads(content)


def recreate_frame(frame: List[List], L: float, grid_step: float) -> Sky:
    sky = Sky(L, grid_step)
    for bird in frame:
        pos, angle, vel, ang_vel = bird
        sky.add_bird(Bird(np.array(pos), vel, ang_vel, angle))

    return sky
