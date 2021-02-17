from typing import Union
from typing import Optional

from .typing import Literal
from .typing import RandomStateType
from .typing import ArrayLike


class DummyClassifier:
    def __init__(
        self,
        strategy: Literal[
            "stratified", "most_frequent", "prior", "uniform", "constant", "warn"
        ] = "warn",
        random_state: RandomStateType = None,
        constant: Union[int, str, ArrayLike, None] = None,
    ):
        ...


class DummyRegressor:
    def __init__(
        self,
        strategy: Literal["mean", "median", "quantile", "constant"] = "mean",
        constant: Union[int, float, ArrayLike, None] = None,
        quantile: Optional[float] = None,
    ):
        ...
