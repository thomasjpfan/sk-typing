from typing import Literal
from typing import Optional

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from ._typing import CVType


class CalibratedClassifierCVAnnotation:
    __estimator__ = CalibratedClassifierCV

    def __init__(
        self,
        base_estimator: BaseEstimator = None,
        *,
        method: Literal["sigmoid", "isotonic"] = "sigmoid",
        cv: CVType = None,
        n_jobs: Optional[int] = None,
        ensemble: bool = True,
    ):
        pass
