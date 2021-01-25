from importlib import import_module
from inspect import getmembers

from ._typing import ArrayLike
from ._typing import NDArray
from ._typing import EstimatorType
from ._typing import CVType
from ._typing import RandomStateType
from ._typing import MemoryType

_ALL_ANNOTATIONS = {}

_MODULES = [
    "calibration",
    "cluster",
    "compose",
    "covariance",
    "cross_decomposition",
    "decomposition",
    "discriminant_analysis",
    "dummy",
    "ensemble",
    "feature_extraction",
    "feature_selection",
    "gaussian_process",
    "impute",
]

for modules in _MODULES:
    mod = import_module(f".{modules}", package="sk_typing")
    for _, member in getmembers(mod):
        if not hasattr(member, "__estimator__"):
            continue
        _ALL_ANNOTATIONS[member.__estimator__] = member

__all__ = [
    "get_init_annotations",
    "ArrayLike",
    "NDArray",
    "EstimatorType",
    "CVType",
    "RandomStateType",
    "MemoryType",
]


def get_init_annotations(Estimator):
    """Get init annotations for estimator.

    Parameters
    ----------
    Estimator : estimator class

    Returns
    -------
    annotation: dict
    """
    try:
        return _ALL_ANNOTATIONS[Estimator].__init__.__annotations__
    except KeyError:
        raise ValueError(f"Type annotations was not defined for {Estimator}")
