import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class AabbData:
    frame_number: int
    aabb: np.array


def read_aabbs_csv(input_csv_path: Path) -> List[AabbData]:
    """
    The csv file contains 1 row per frame describing a 3D axis aligned bounding box (aabb) around the object of
    interest (the actor in our case).
    """
    aabbs = []

    with open(input_csv_path, "r", newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            aabbs.append(
                AabbData(
                    frame_number=int(row["frame_number"]),
                    aabb=np.array(
                        [
                            float(row["aabb_min_x"]),
                            float(row["aabb_min_y"]),
                            float(row["aabb_min_z"]),
                            float(row["aabb_max_x"]),
                            float(row["aabb_max_y"]),
                            float(row["aabb_max_z"]),
                        ]
                    ).reshape(2, 3),
                )
            )

    return aabbs


def write_aabbs_csv(aabbs: List[AabbData], output_csv_path: Path) -> List[AabbData]:
    csv_field_names = [
        "frame_number",
        "aabb_min_x",
        "aabb_min_y",
        "aabb_min_z",
        "aabb_max_x",
        "aabb_max_y",
        "aabb_max_z",
    ]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_field_names)
        writer.writeheader()

        for aabb in aabbs:
            csv_row = {}
            csv_row["frame_number"] = str(aabb.frame_number)
            csv_row["aabb_min_x"] = str(aabb.aabb[0, 0])
            csv_row["aabb_min_y"] = str(aabb.aabb[0, 1])
            csv_row["aabb_min_z"] = str(aabb.aabb[0, 2])
            csv_row["aabb_max_x"] = str(aabb.aabb[1, 0])
            csv_row["aabb_max_y"] = str(aabb.aabb[1, 1])
            csv_row["aabb_max_z"] = str(aabb.aabb[1, 2])

            assert len(csv_row) == len(csv_field_names)
            writer.writerow(csv_row)
