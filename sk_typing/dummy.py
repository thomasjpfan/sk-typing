from typing import Union
from typing import Optional

from sklearn.dummy import DummyRegressor
from sklearn.dummy import DummyClassifier

from ._typing import Literal
from ._typing import RandomState
from ._typing import ArrayLike


class DummyClassifierAnnotation:
    __estimator__ = DummyClassifier

    def __init__(
        self,
        strategy: Literal[
            "stratified", "most_frequent", "prior", "uniform", "constant"
        ] = "prior",
        random_state: RandomState = None,
        constant: Union[int, str, ArrayLike, None] = None,
    ):
        pass


class DummyRegressorAnnotation:
    __estimator__ = DummyRegressor

    def __init__(
        self,
        strategy: Literal["mean", "median", "quantile", "constant"] = "mean",
        constant: Union[int, float, ArrayLike, None] = None,
        quantile: Optional[float] = None,
    ):
        pass
