from typing import Union
from typing import Optional

import pytest
from typing_extensions import Literal

from sk_typing.convert import get_d3m_representation


@pytest.mark.parametrize("annotation", [bool, int, float, str])
def test_get_d3m_representation_builtin(annotation):
    output = get_d3m_representation("parameter", annotation)
    assert output["type"] == "Hyperparameter"
    assert output["name"] == "parameter"
    assert output["init_args"]["semantic_types"][0] == "parameter"
    assert output["init_args"]["_structural_type"] == annotation.__name__


@pytest.mark.parametrize(
    "annotation, default",
    [(bool, True), (int, 10), (float, 0.1), (str, "Hello world")],
)
def test_get_d3m_representation_builtin_default(annotation, default):
    output = get_d3m_representation(
        "parameter", annotation, description="This is awesome", default=default
    )
    if annotation == bool:
        assert output["init_args"]["default"] == str(default)
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
