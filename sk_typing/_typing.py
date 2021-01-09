import numpy as np
from joblib import Memory

from typing import Union
from typing import Protocol
from typing import Iterator
from typing import Literal
from typing import Tuple
from typing import TypeVar


class CVSplitter(Protocol):
    def get_n_splits(self):
        ...

    def split(self, X, y=None, groups=None):
        ...


CVType = Union[
    int, CVSplitter, Iterator[Tuple[np.array, np.ndarray]], Literal["prefit"], None
]

ArrayLike = TypeVar("ArrayLike")
BaseEstimatorType = TypeVar("BaseEstimatorType")
RandomState = Union[int, np.random.RandomState, None]
MemroyType = Union[str, Memory, None]
