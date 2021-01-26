import numpy as np
from typing import Optional
from typing import Union
from typing import Callable

from ._typing import EstimatorType
from ._typing import Literal
from ._typing import RandomStateType
from ._typing import CVType

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV


class GridSearchCVAnnotation:
    __estimator__ = GridSearchCV

    def __init__(
        self,
        estimator: EstimatorType,
        param_grid: Union[dict, list],
        scoring: Union[str, Callable, list, tuple, dict, None] = None,
        n_jobs: Optional[int] = None,
        refit: Union[bool, str, Callable] = True,
        cv: CVType = None,
        verbose: int = 0,
        pre_dispatch: str = "2*n_jobs",
        error_score: Union[Literal["raise"], float] = np.nan,
        return_train_score: bool = False,
    ):
        pass


class HalvingGridSearchCVAnnotation:
    __estimator__ = HalvingGridSearchCV

    def __init__(
        self,
        estimator: EstimatorType,
        param_grid: Union[dict, list],
        factor: Union[int, float] = 3,
        resource: str = "n_samples",
        max_resources: Union[Literal["auto"], int] = "auto",
        min_resources: Union[Literal["exhaust", "smallest"], int] = "exhaust",
        aggressive_elimination: bool = False,
        cv: CVType = 5,
        scoring: Union[str, Callable, None] = None,
        refit: bool = True,
        error_score: Union[Literal["raise"], float] = np.nan,
        return_train_score: bool = True,
        random_state: RandomStateType = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        pass


class HalvingRandomSearchCVAnnotation:
    __estimator__ = HalvingRandomSearchCV

    def __init__(
        self,
        estimator: EstimatorType,
        param_distributions: dict,
        n_candidates: Union[Literal["exhaust"], int] = "exhaust",
        factor: Union[int, float] = 3,
        resource: str = "n_samples",
        max_resources: Union[Literal["auto"], int] = "auto",
        min_resources: Union[Literal["exhaust", "smallest"], int] = "smallest",
        aggressive_elimination: bool = False,
        cv: CVType = 5,
        scoring: Union[str, Callable, None] = None,
        refit: bool = True,
        error_score: Union[Literal["raise"], float] = np.nan,
        return_train_score: bool = True,
        random_state: RandomStateType = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        pass


class RandomizedSearchCVAnnotation:
    __estimator__ = RandomizedSearchCV

    def __init__(
        self,
        estimator: EstimatorType,
        param_distributions: Union[dict, list],
        n_iter: int = 10,
        scoring: Union[str, Callable, list, tuple, dict, None] = None,
        n_jobs: Optional[int] = None,
        refit: Union[bool, str, Callable] = True,
        cv: CVType = None,
        verbose: int = 0,
        pre_dispatch: str = "2*n_jobs",
        random_state: RandomStateType = None,
        error_score: Union[Literal["raise"], float] = np.nan,
        return_train_score: bool = False,
    ):
        pass
