from typing import Optional
from typing import Union

from .typing import CVType
from .typing import EstimatorType
from .typing import Literal
from .typing import NDArray


class CalibratedClassifierCV:
    classes_: NDArray
    calibrated_classifiers_: list

    def __init__(
        self,
        base_estimator: EstimatorType = None,
        method: Literal["sigmoid", "isotonic"] = "sigmoid",
        cv: Union[CVType, Literal["prefit"]] = None,
        n_jobs: Optional[int] = None,
        ensemble: bool = True,
    ):
        ...
