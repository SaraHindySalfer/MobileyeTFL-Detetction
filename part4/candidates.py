from dataclasses import dataclass

import numpy as np


@dataclass
class Candidates:
    frame_path: str
    points: list
    auxiliary: np.ndarray