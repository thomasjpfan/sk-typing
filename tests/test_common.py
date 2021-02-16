from inspect import signature
from numpydoc.docscrape import ClassDoc

import pytest
from sklearn.utils import all_estimators
from numpy.testing import assert_allclose

from sk_typing import get_metadata
from sk_typing import _ALL_ANNOTATIONS

ESTIMATORS_TO_CHECK = [est for _, est in all_estimators()]

MODULES_TO_SKIP = {
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
}


@pytest.mark.parametrize(
    "Estimator",
    [
        est
        for est in ESTIMATORS_TO_CHECK
        if est.__module__.split(".")[1] not in MODULES_TO_SKIP
    ],
)
def test_get_metadata_attributes(Estimator):
    """Check attributes are defined."""
    meta_data = get_metadata(Estimator.__name__)
    annotated_attr = set(meta_data["attributes"])

    class_doc = ClassDoc(Estimator)
    docstring_attributes = set(p.name for p in class_doc["Attributes"])

    assert annotated_attr <= docstring_attributes


@pytest.mark.parametrize("Estimator", ESTIMATORS_TO_CHECK, ids=lambda est: est.__name__)
def test_get_metadata(Estimator):
    """Check metadat are consistant with the estimator."""
    # check annotation are consistent with the values in `__init__`
    # check docstring parameters for basic types

    meta_data = get_metadata(Estimator.__name__)
    init_annotations = meta_data["parameters"]
    init_parameters = signature(Estimator).parameters

    assert set(init_annotations) == set(init_parameters)

    # All hyperparameters are annotated
    for p in init_parameters:
        assert p in init_annotations


@pytest.mark.parametrize("Estimator", ESTIMATORS_TO_CHECK, ids=lambda est: est.__name__)
def test_annotation_object(Estimator):
    """Check annotation."""
    annotation_obj = _ALL_ANNOTATIONS[Estimator.__name__]
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
