from inspect import signature

import pytest
from sklearn.utils import all_estimators
from numpy.testing import assert_allclose

from sk_typing import get_init_annotations
from sk_typing import _ALL_ANNOTATIONS

MODULES_TO_IGNORE = {
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
}
ESTIMATORS_TO_CHECK = [
    est
    for _, est in all_estimators()
    if est.__module__.split(".")[1] not in MODULES_TO_IGNORE
]


@pytest.mark.parametrize("Estimator", ESTIMATORS_TO_CHECK, ids=lambda est: est.__name__)
def test_init_annotations(Estimator):
    """Check init annotations are consistant with the estimator."""
    # check annotation are consistent with the values in `__init__`
    # check docstring parameters for basic types

    init_annotations = get_init_annotations(Estimator)
    init_parameters = signature(Estimator).parameters

    assert set(init_annotations) == set(init_parameters)

    # All hyperparameters are annotated
    for p in init_parameters:
        assert p in init_annotations


@pytest.mark.parametrize("Estimator", ESTIMATORS_TO_CHECK, ids=lambda est: est.__name__)
def test_annotation_object(Estimator):
    """Check annotation objects."""
    annotation_obj = _ALL_ANNOTATIONS[Estimator]
    obj_params = signature(annotation_obj).parameters
    init_params = signature(Estimator).parameters

    assert set(obj_params) == set(init_params)

    for key, param in obj_params.items():
        obj_default = param.default
        init_default = init_params[key].default

        if isinstance(obj_default, float):
            assert_allclose(init_default, obj_default)
        else:
            assert init_params[key].default == param.default
