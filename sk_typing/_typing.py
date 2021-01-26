import numpy as np
from joblib import Memory

import typing
from typing import Union
from typing import Iterator
from typing import Tuple
from typing import TypeVar

try:
    import typing_extensions  # noqa

    TYPING_EXTENSION_INSTALLED = True
except ImportError:
    TYPING_EXTENSION_INSTALLED = False


if typing.TYPE_CHECKING or TYPING_EXTENSION_INSTALLED:
    from typing_extensions import Protocol  # noqa

    class CVSplitter(Protocol):
        def get_n_splits(self):
            """Get the number of splits."""

        def split(self, X, y=None, groups=None):
            """Split data"""


else:
    CVSplitter = TypeVar("CVSplitter")  # typing: ignore

if typing.TYPE_CHECKING or TYPING_EXTENSION_INSTALLED:
    from typing_extensions import Literal  # noqa
else:

    class _SimpleLiteral:
        def __getitem__(self, values):
            return typing.Any

    Literal = _SimpleLiteral()


CVType = Union[int, CVSplitter, Iterator[Tuple[np.array, np.ndarray]], None]

ArrayLike = TypeVar("ArrayLike")
NDArray = TypeVar("NDArray")
EstimatorType = TypeVar("EstimatorType")
DType = TypeVar("DType")
KernelType = TypeVar("KernelType")
RandomStateType = Union[int, np.random.RandomState, None]
MemoryType = Union[str, Memory, None]
