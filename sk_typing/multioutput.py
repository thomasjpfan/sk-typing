from typing import Optional
from typing import Union

from ._typing import EstimatorType
from ._typing import Literal
from ._typing import RandomStateType
from ._typing import ArrayLike
from ._typing import CVType

from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import RegressorChain


class ClassifierChainAnnotation:
    __estimator__ = ClassifierChain

    def __init__(
        self,
        base_estimator: EstimatorType,
        order: Union[Literal["random"], ArrayLike, None] = None,
        cv: CVType = None,
        random_state: RandomStateType = None,
    ):
        pass


class MultiOutputClassifierAnnotation:
    __estimator__ = MultiOutputClassifier

    def __init__(self, estimator: EstimatorType, n_jobs: Optional[int] = None):
        pass


class MultiOutputRegressorAnnotation:
    __estimator__ = MultiOutputRegressor

    def __init__(self, estimator: EstimatorType, n_jobs: Optional[int] = None):
        pass


class RegressorChainAnnotation:
    __estimator__ = RegressorChain

    def __init__(
        self,
        base_estimator: EstimatorType,
        order: Union[ArrayLike, Literal["random"], None] = None,
        cv: CVType = None,
        random_state: RandomStateType = None,
    ):
        pass
