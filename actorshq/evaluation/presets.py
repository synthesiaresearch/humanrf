import itertools
from typing import List, Tuple

import numpy as np

# fmt: off
# Here camera indices are 0-based (which differs to their name in the dataset, which is 1-indexed)
camera_configs = {

    "siggraph_train": (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69, 71,
        72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 105,
        106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 118, 119, 120, 121, 122, 123, 124, 125, 127, 130, 131, 132,
        133, 134, 135, 138, 139, 140, 141, 142, 143, 148, 149, 150, 151, 156, 157, 158, 159,
    ),
    "siggraph_train_validation": (
        # all landscape
        10, 19, 33, 44, 50, 73, 83, 90, 104, 117,
    ),
    "siggraph_test": (
        # landscape
        0, 13, 24, 30, 43, 57, 63, 64, 70, 84, 97, 103, 114,
        # portrait
        126,  # We only use the hero portrait camera.
    ),
    "siggraph_vmaf": (126,),
}
# fmt: on


assert len(set(camera_configs["siggraph_test"])) == len(camera_configs["siggraph_test"])
assert len(set(camera_configs["siggraph_train_validation"])) == len(camera_configs["siggraph_train_validation"])
assert len(set(camera_configs["siggraph_train"])) == len(camera_configs["siggraph_train"])
assert len(set(camera_configs["siggraph_vmaf"])) == len(camera_configs["siggraph_vmaf"])


# The first index here is inclusive, while the second index is exclusive (like `range(start, end)`)
frame_configs = {
    "siggraph_interval_0": (15, 15 + 20),
    "siggraph_interval_1": (15, 15 + 50),
    "siggraph_interval_2": (15, 15 + 100),
    "siggraph_interval_3": (15, 15 + 250),
    "siggraph_interval_4": (15, 15 + 500),
    "siggraph_interval_5": (15, 15 + 1000),
}


def get_spaced_elements(array, count):
    return [array[i] for i in np.round(np.linspace(0, len(array) - 1, count)).astype(int)]


def get_vmaf_test_sequence(frame_numbers: List[int]):
    assert len(camera_configs["siggraph_vmaf"]) == 1
    return list(zip(itertools.repeat(camera_configs["siggraph_vmaf"][0]), frame_numbers[::3]))  # Every 3rd frame.


def get_render_sequence(
    coverage: str,
    camera_preset: str,
    frame_numbers: List[int],
    repeat_cameras: int = 1,
    repeat_frames: int = 1,
) -> List[Tuple[int, int]]:
    camera_numbers = list(
        itertools.chain.from_iterable(itertools.repeat(camera_configs[camera_preset], repeat_cameras))
    )
    frame_numbers = list(itertools.chain.from_iterable(itertools.repeat(frame_numbers, repeat_frames)))

    if coverage == "siggraph_test":
        assert camera_preset == "siggraph_test"
        render_sequence_hero = get_vmaf_test_sequence(frame_numbers)
        landscape_views = [0, 63, 97, 30, 13, 70, 114, 24, 84, 43, 64, 103, 57]
        render_sequence_landscape_complete = [
            (landscape_views[i % len(landscape_views)], frame_idx)
            for i, frame_idx in enumerate(frame_numbers[::5])  # Every 5th frame.
        ]
        return list(set(render_sequence_hero + render_sequence_landscape_complete))

    if coverage == "exhaustive":
        return list(itertools.product(camera_numbers, frame_numbers))

    if coverage == "uniform":
        return list(zip(camera_numbers, get_spaced_elements(frame_numbers, len(camera_numbers))))

    raise NotImplementedError()
