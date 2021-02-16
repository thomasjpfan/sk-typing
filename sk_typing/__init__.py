from importlib import import_module
from inspect import getmembers, isclass

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
    "isotonic",
    "kernel_approximation",
    "kernel_ridge",
    "linear_model",
    "manifold",
    "mixture",
    "model_selection",
    "multiclass",
    "multioutput",
    "naive_bayes",
    "neighbors",
    "neural_network",
    "pipeline",
    "preprocessing",
    "random_projection",
    "semi_supervised",
    "svm",
    "tree",
]

for modules in _MODULES:
    mod = import_module(f".{modules}", package="sk_typing")
    for name, member in getmembers(mod, isclass):
        _ALL_ANNOTATIONS[name] = member

__all__ = [
    "get_metadata",
]


def get_metadata(Estimator):
    """Get init annotations for estimator.

    Parameters
    ----------
    Estimator : estimator class

    Returns
    -------
    metadata: dict
    """
    try:
        return {"init": _ALL_ANNOTATIONS[Estimator.__name__].__init__.__annotations__}
    except KeyError:
        raise ValueError(f"Type annotations was not defined for {Estimator}")
