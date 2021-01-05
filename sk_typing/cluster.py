from typing import Union
from typing import Literal

from sklearn.cluster import AffinityPropagation
from ._typing import ArrayLike
from ._typing import RandomState


class AffinityPropagationAnnotation:
    __estimator__ = AffinityPropagation

    def __init__(
        self,
        *,
        damping: float = 0.5,
        max_iter: int = 200,
        convergence_iter: int = 15,
        copy: bool = True,
        preference: Union[float, ArrayLike, None] = None,
        affinity: Literal["euclidean", "precomputed"] = "euclidean",
        verbose: bool = False,
        random_state: Union[RandomState, Literal["warn"]] = "warn"
    ):
        pass
