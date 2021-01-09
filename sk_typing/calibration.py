from typing import Literal
from typing import Optional

from sklearn.calibration import CalibratedClassifierCV
from ._typing import CVType
from ._typing import BaseEstimatorType


class CalibratedClassifierCVAnnotation:
    __estimator__ = CalibratedClassifierCV

    def __init__(
        self,
        base_estimator: BaseEstimatorType = None,
        *,
        method: Literal["sigmoid", "isotonic"] = "sigmoid",
        cv: CVType = None,
        n_jobs: Optional[int] = None,
        ensemble: bool = True,
    ):
        pass
