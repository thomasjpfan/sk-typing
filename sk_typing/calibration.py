import numpy as np

from typing import Union

from .typing import CVType
from .typing import EstimatorType
from .typing import Literal


class CalibratedClassifierCV:
    classes_: np.ndarray
    calibrated_classifiers_: list

    def __init__(
        self,
        base_estimator: EstimatorType = None,
        method: Literal["sigmoid", "isotonic"] = "sigmoid",
        cv: Union[CVType, Literal["prefit"]] = None,
    ):
        ...
