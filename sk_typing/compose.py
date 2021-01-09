from typing import Union
from typing import Literal
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor

from ._typing import BaseEstimatorType


class ColumnTransformerAnnotation:
    __estimator__ = ColumnTransformer

    def __init__(
        self,
        transformers: list,
        remainder: Union[Literal["drop", "passthrough"], BaseEstimatorType] = "drop",
        sparse_threshold: float = 0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
    ):
        pass


class TransformedTargetRegressorAnnotation:
    __estimator__ = TransformedTargetRegressor

    def __init__(
        self,
        regressor=None,
        transformer=None,
        func=None,
        inverse_func=None,
        check_inverse=True,
    ):
        pass
