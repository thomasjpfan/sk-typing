import numpy as np

from typing import Union
from typing import Protocol
from typing import Iterator
from typing import Literal
from typing import Tuple


class CVSplitter(Protocol):
    def get_n_splits(self):
        ...

    def split(self, X, y=None, groups=None):
        ...


CV = Union[
    int, CVSplitter, Iterator[Tuple[np.array, np.array]], Literal["prefit"], None
]
