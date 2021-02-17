from typing import Union
from typing import Optional
from collections.abc import Callable

from .typing import EstimatorType
from .typing import Literal


class ColumnTransformer:
    transformers_: list
    named_transformers_: dict
    sparse_output_: bool

    def __init__(
        self,
        transformers: list,
        remainder: Union[Literal["drop", "passthrough"], EstimatorType] = "drop",
        sparse_threshold: float = 0.3,
        n_jobs: Optional[int] = None,
        transformer_weights: Optional[dict] = None,
        verbose: bool = False,
    ):
        ...


class TransformedTargetRegressor:
    regressor_: EstimatorType
    transformer_: object

    def __init__(
        self,
        regressor: Optional[EstimatorType] = None,
        transformer: Optional[EstimatorType] = None,
        func: Callable = None,
        inverse_func: Callable = None,
        check_inverse: bool = True,
    ):
        ...
