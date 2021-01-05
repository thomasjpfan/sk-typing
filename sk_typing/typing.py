import numpy as np

from typing import Union
from typing import Protocol
from typing import Iterator
from typing import Literal


class CVSplitter(Protocol):
    def get_n_splits(self):
        ...

    def split(self, X, y=None, groups=None):
        ...


CV = Union[int, CVSplitter, Iterator[np.array, np.array], Literal["prefit"], None]
