from typing import Optional
from typing import Union
from typing import Callable

from .typing import EstimatorType
from .typing import Literal
from .typing import RandomStateType

import numpy as np


class IterativeImputer:
    def __init__(
        self,
        estimator: Optional[EstimatorType] = None,
        missing_values: Union[int, float] = np.nan,
        sample_posterior: bool = False,
        max_iter: int = 10,
        tol: float = 0.001,
        n_nearest_features: Optional[int] = None,
        initial_strategy: Literal[
            "mean", "median", "most_frequent", "constant"
        ] = "mean",
        imputation_order: Literal[
            "ascending", "descending", "roman", "arabic", "random"
        ] = "ascending",
        skip_complete: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        verbose: int = 0,
        random_state: RandomStateType = None,
        add_indicator: bool = False,
    ):
        ...


class KNNImputer:
    def __init__(
        self,
        missing_values: Union[int, float, str, None] = np.nan,
        n_neighbors: int = 5,
        weights: Union[Literal["uniform", "distance"], Callable] = "uniform",
        metric: Union[Literal["nan_euclidean"], Callable] = "nan_euclidean",
        copy: bool = True,
        add_indicator: bool = False,
    ):
        ...


class MissingIndicator:
    def __init__(
        self,
        missing_values: Union[int, float, str, None] = np.nan,
        features: Literal["missing-only", "all"] = "missing-only",
        sparse: Union[bool, Literal["auto"]] = "auto",
        error_on_new: bool = True,
    ):
        ...


class SimpleImputer:
    def __init__(
        self,
        missing_values: Union[int, float, str, None] = np.nan,
        strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
        fill_value: Union[str, float, int, None] = None,
        verbose: int = 0,
        copy: bool = True,
        add_indicator: bool = False,
    ):
        ...
