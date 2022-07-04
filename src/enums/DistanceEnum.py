from enum import Enum
from functools import partial
from src.utils import DistanceUtils


class DistMethod(Enum):
    EUCLIDEAN = partial(DistanceUtils.get_euclidean_dist)
    MANHATTAN = partial(DistanceUtils.get_manhattan_dist)
