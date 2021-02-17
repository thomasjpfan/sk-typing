from typing import Union
from typing import Optional

from .typing import Literal
from .typing import ArrayLike


class LinearDiscriminantAnalysis:
    def __init__(
        self,
        solver: Literal["svd", "lsqr", "eigen"] = "svd",
        shrinkage: Union[Literal["auto"], float, None] = None,
        priors: Optional[ArrayLike] = None,
        n_components: Optional[int] = None,
        store_covariance: bool = False,
        tol: float = 0.0001,
    ):
        ...


class QuadraticDiscriminantAnalysis:
    def __init__(
        self,
        priors: Optional[ArrayLike] = None,
        reg_param: float = 0.0,
        store_covariance: bool = False,
        tol: float = 0.0001,
    ):
        ...
