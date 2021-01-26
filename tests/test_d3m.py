from typing import Union
from typing import Optional
import json

import pytest

from sklearn.utils import all_estimators
from sk_typing.convert.d3m import _get_output_for_estimator

from typing_extensions import Literal

from sk_typing._typing import ArrayLike
from sk_typing._typing import NDArray
from sk_typing._typing import EstimatorType


from sk_typing.convert import get_d3m_representation


ALL_ESTIMATORS = all_estimators()
MODULES_TO_IGNORE = {
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
    (name, est)
    for name, est in ALL_ESTIMATORS
    if est.__module__.split(".")[1] not in MODULES_TO_IGNORE
]


@pytest.mark.parametrize("name, estimator", ESTIMATORS_TO_CHECK)
def test_get_output_for_module(name, estimator):
    """Smoke test for modules"""
    output = _get_output_for_estimator(name, estimator)
    json.dumps(output)


@pytest.mark.parametrize("annotation", [bool, int, float, str, tuple])
def test_get_d3m_representation_builtin(annotation):
    output = get_d3m_representation("parameter", annotation)
    assert output["type"] == "Hyperparameter"
    assert output["name"] == "parameter"
    assert output["init_args"]["semantic_types"][0] == "parameter"
    assert output["init_args"]["_structural_type"] == annotation.__name__


@pytest.mark.parametrize(
    "annotation, default",
    [(bool, True), (int, 10), (float, 0.1), (str, "Hello world"), (tuple, (1, 2))],
)
def test_get_d3m_representation_builtin_default(annotation, default):
    output = get_d3m_representation(
        "parameter", annotation, description="This is awesome", default=default
    )
    if annotation == bool:
        assert output["init_args"]["default"] == str(default)
    elif annotation == tuple:
        assert output["init_args"]["default"] == f"&esc{default}"
    else:
        assert output["init_args"]["default"] == default
    assert output["init_args"]["description"] == "This is awesome"


def test_get_d3m_representation_builtin_none():
    output = get_d3m_representation("this_is_none", None, default="None")
    assert output["name"] == "this_is_none"
    assert output["type"] == "Constant"
    assert output["init_args"]["default"] == "None"
    assert output["init_args"]["_structural_type"] == "None"


def test_get_d3m_representation_literal():
    literal_dtype = Literal["auto", "sqrt", "log2"]
    output = get_d3m_representation("calculated", literal_dtype, default="auto")
    assert output["name"] == "calculated"
    assert output["type"] == "Enumeration"
    assert output["init_args"]["values"] == ["auto", "sqrt", "log2"]
    assert output["init_args"]["_structural_type"] == "str"
    assert output["init_args"]["default"] == "auto"
    assert output["init_args"]["semantic_types"] == ["calculated"]


def test_test_get_d3m_representation_literal_single():
    single_dtype = Literal["auto"]
    output = get_d3m_representation("hello", single_dtype, default="auto")
    assert output["name"] == "hello"
    assert output["type"] == "Constant"
    assert output["init_args"]["default"] == "auto"
    assert output["init_args"]["_structural_type"] == "str"


def test_get_d3m_representation_union():
    simple_union = Union[Literal["auto", "sqrt", "log2"], int, float]
    output = get_d3m_representation("max_features", simple_union, default=1)
    assert output["name"] == "max_features"
    assert output["type"] == "Union"
    assert output["init_args"]["default"] == "max_features__int"
    assert output["init_args"]["semantic_types"] == ["max_features"]

    hyperparams = output["hyperparams"]
    assert len(hyperparams) == 3

    literal_output = hyperparams[0]
    assert literal_output["name"] == "max_features__str"
    assert literal_output["type"] == "Enumeration"
    assert literal_output["init_args"]["values"] == ["auto", "sqrt", "log2"]
    assert literal_output["init_args"]["_structural_type"] == "str"

    int_output = hyperparams[1]
    assert int_output["name"] == "max_features__int"
    assert int_output["type"] == "Hyperparameter"
    assert int_output["init_args"]["_structural_type"] == "int"
    assert int_output["init_args"]["default"] == 1

    float_output = hyperparams[2]
    assert float_output["name"] == "max_features__float"
    assert float_output["type"] == "Hyperparameter"
    assert float_output["init_args"]["_structural_type"] == "float"


def test_get_d3m_representation_union_one_literal():
    simple_union = Union[Literal["auto"], float]
    output = get_d3m_representation("max_depth", simple_union, default="auto")
    assert output["name"] == "max_depth"
    assert output["type"] == "Union"
    assert output["init_args"]["default"] == "max_depth__str"
    assert output["init_args"]["semantic_types"] == ["max_depth"]

    hyperparams = output["hyperparams"]
    assert len(hyperparams) == 2

    literal_output = hyperparams[0]
    assert literal_output["name"] == "max_depth__str"
    assert literal_output["type"] == "Constant"
    assert literal_output["init_args"]["default"] == "auto"
    assert literal_output["init_args"]["_structural_type"] == "str"

    float_output = hyperparams[1]
    assert float_output["name"] == "max_depth__float"
    assert float_output["type"] == "Hyperparameter"
    assert float_output["init_args"]["_structural_type"] == "float"


def test_get_d3m_representation_union_none():
    none_union = Union[float, int, None]
    output = get_d3m_representation("great_union", none_union, default=None)
    assert output["name"] == "great_union"
    assert output["type"] == "Union"
    assert output["init_args"]["default"] == "great_union__None"
    assert output["init_args"]["semantic_types"] == ["great_union"]

    hyperparams = output["hyperparams"]
    assert len(hyperparams) == 3

    float_output = hyperparams[0]
    assert float_output["name"] == "great_union__float"
    assert float_output["type"] == "Hyperparameter"
    assert float_output["init_args"]["_structural_type"] == "float"

    int_output = hyperparams[1]
    assert int_output["name"] == "great_union__int"
    assert int_output["type"] == "Hyperparameter"
    assert int_output["init_args"]["_structural_type"] == "int"

    none_output = hyperparams[2]
    assert none_output["name"] == "great_union__None"
    assert none_output["type"] == "Constant"
    assert none_output["init_args"]["default"] == "None"
    assert none_output["init_args"]["_structural_type"] == "None"


@pytest.mark.parametrize("annotation", [Optional[int], Union[int, None]])
def test_get_d3m_representation_optional(annotation):
    output = get_d3m_representation("n_jobs", annotation, default=1)
    assert output["name"] == "n_jobs"
    assert output["type"] == "Union"
    assert output["init_args"]["default"] == "n_jobs__int"
    assert output["init_args"]["semantic_types"] == ["n_jobs"]

    hyperparams = output["hyperparams"]
    assert len(hyperparams) == 2

    int_output = hyperparams[0]
    assert int_output["name"] == "n_jobs__int"
    assert int_output["type"] == "Hyperparameter"
    assert int_output["init_args"]["_structural_type"] == "int"
    assert int_output["init_args"]["default"] == 1

    none_output = hyperparams[1]
    assert none_output["name"] == "n_jobs__None"
    assert none_output["type"] == "Constant"
    assert none_output["init_args"]["_structural_type"] == "None"


@pytest.mark.parametrize("array_like", [ArrayLike, NDArray])
def test_get_d3m_representation_ndarray(array_like):
    output = get_d3m_representation("X", Optional[array_like], default=None)
    assert output["name"] == "X"
    assert output["type"] == "Union"
    assert output["init_args"]["default"] == "X__None"

    hyperparams = output["hyperparams"]
    assert len(hyperparams) == 2

    array_output = hyperparams[0]
    assert array_output["name"] == "X__ndarray"
    assert array_output["type"] == "Hyperparameter"
    assert array_output["init_args"]["_structural_type"] == "ndarray"

    none_output = hyperparams[1]
    assert none_output["name"] == "X__None"
    assert none_output["type"] == "Constant"
    assert none_output["init_args"]["_structural_type"] == "None"


def test_get_d3m_representation_base_estimator():
    output = get_d3m_representation("base_estimator", EstimatorType)
    assert output["name"] == "base_estimator"
    assert output["type"] == "Hyperparameter"
    assert output["init_args"]["_structural_type"] == "Estimator"
