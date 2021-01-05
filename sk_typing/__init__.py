from importlib import import_module

_ALL_ANNOTATIONS = {}

_MODULES = ["calibration"]

for modules in _MODULES:
    mod = import_module(f".{modules}", package="sk_typing")
    for annotation in mod.annotations:
        _ALL_ANNOTATIONS[annotation.__estimator__] = annotation

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
