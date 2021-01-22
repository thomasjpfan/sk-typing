from importlib import import_module
from inspect import getmembers

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
]

for modules in _MODULES:
    mod = import_module(f".{modules}", package="sk_typing")
    for _, member in getmembers(mod):
        if not hasattr(member, "__estimator__"):
            continue
        _ALL_ANNOTATIONS[member.__estimator__] = member

__all__ = ["get_init_annotations"]


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
