from typing import Union
from typing import Optional

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ._typing import Literal
from ._typing import ArrayLike
from ._typing import EstimatorType


class LinearDiscriminantAnalysisAnnotation:
    __estimator__ = LinearDiscriminantAnalysis

    def __init__(
        self,
        solver: Literal["svd", "lsqr", "eigen"] = "svd",
        shrinkage: Union[Literal["auto"], float, None] = None,
        priors: Optional[ArrayLike] = None,
        n_components: Optional[int] = None,
        store_covariance: bool = False,
        tol: float = 0.0001,
        covariance_estimator: Optional[EstimatorType] = None,
    ):
        pass


class QuadraticDiscriminantAnalysisAnnotation:
    __estimator__ = QuadraticDiscriminantAnalysis

    def __init__(
        self,
        priors: Optional[ArrayLike] = None,
        reg_param: float = 0.0,
        store_covariance: bool = False,
        tol: float = 0.0001,
    ):
        pass
