from typing import Optional
from typing import Union

from sklearn.calibration import CalibratedClassifierCV
from ._typing import CVType
from ._typing import BaseEstimatorType
from ._typing import Literal


class CalibratedClassifierCVAnnotation:
    __estimator__ = CalibratedClassifierCV

    def __init__(
        self,
        base_estimator: BaseEstimatorType = None,
        *,
        method: Literal["sigmoid", "isotonic"] = "sigmoid",
        cv: Union[CVType, Literal["prefit"]] = None,
        n_jobs: Optional[int] = None,
        ensemble: bool = True,
    ):
        pass
