import numpy as np
from joblib import Memory

import typing
from typing import Union
from typing import Iterator
from typing import Tuple
from typing import TypeVar
from typing import Any


if hasattr(typing, "Protocol"):
    from typing import Protocol  # noqa
elif typing.TYPE_CHECKING:
    from typing_extensions import Protocol  # noqa
else:
    Protocol = Any

if hasattr(typing, "Literal"):
    from typing import Literal  # noqa
elif typing.TYPE_CHECKING:
    from typing_extensions import Literal  # noqa
else:

    class _SimpleLiteral:
        def __getitem__(self, values):
            return typing.Any

    Literal = _SimpleLiteral()


class CVSplitter(Protocol):
    def get_n_splits(self):
        ...

    def split(self, X, y=None, groups=None):
        ...


CVType = Union[
    int, CVSplitter, Iterator[Tuple[np.array, np.ndarray]], Literal["prefit"], None
]

ArrayLike = TypeVar("ArrayLike")
NDArray = TypeVar("NDArray")
BaseEstimatorType = TypeVar("BaseEstimatorType")
RandomState = Union[int, np.random.RandomState, None]
MemroyType = Union[str, Memory, None]
