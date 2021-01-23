from typing import Union
from typing import Optional
from collections.abc import Callable
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor

from ._typing import EstimatorType
from ._typing import Literal


class ColumnTransformerAnnotation:
    __estimator__ = ColumnTransformer

    def __init__(
        self,
        transformers: list,
        remainder: Union[Literal["drop", "passthrough"], EstimatorType] = "drop",
        sparse_threshold: float = 0.3,
        n_jobs: Optional[int] = None,
        transformer_weights: Optional[dict] = None,
        verbose: bool = False,
    ):
        pass


class TransformedTargetRegressorAnnotation:
    __estimator__ = TransformedTargetRegressor

    def __init__(
        self,
        regressor: Optional[EstimatorType] = None,
        transformer: Optional[EstimatorType] = None,
        func: Callable = None,
        inverse_func: Callable = None,
        check_inverse: bool = True,
    ):
        pass
