import inspect
import numpy as np
from ._extract import unpack_annotation
from ._extract import AnnotatedMeta
from ..typing import EstimatorType


def _get_default(default):
    if isinstance(default, bool):
        return str(default)
    elif isinstance(default, tuple):
        return f"&esc{default}"
    elif default is None:
        return "None"
    else:
        return default


def _is_instance(value, annotation):
    annotation_meta = unpack_annotation(annotation)
    if annotation_meta.class_name in {"bool", "int", "float", "str", "list", "dict"}:
        return isinstance(value, annotation)
    elif annotation_meta.class_name == "Literal":
        for item in annotation_meta.args:
            if value == item:
                return True
        return False
    elif annotation_meta.class_name == "None":
        return value is None
    elif annotation_meta.class_name == "Callable":
        return callable(value)
    else:
        ValueError(f"Unsupported annotation: {annotation}")


def _process_builtins(name, annotation_meta):
    return {
        "type": "Hyperparameter",
        "name": name,
        "init_args": {
            "semantic_types": [name],
            "_structural_type": annotation_meta.class_name,
        },
    }


def _process_literal(name, annotation_meta):
    values = list(annotation_meta.args)
    if len(values) == 1:
        # Literal are all strings
        new_meta = AnnotatedMeta(
            class_name=type(values[0]).__name__, args=(), metadata=()
        )
        return _process_constant(name, new_meta)

    return {
        "type": "Enumeration",
        "name": name,
        "init_args": {
            "semantic_types": [name],
            "values": values,
            "_structural_type": "str",
        },
    }


def _process_constant(name, annotation_meta):
    return {
        "type": "Constant",
        "name": name,
        "init_args": {
            "semantic_types": [name],
            "_structural_type": annotation_meta.class_name,
        },
    }


def _process_union(name, annotation_meta, default=inspect.Parameter.empty):
    output = {"name": name, "type": "Union", "init_args": {"semantic_types": [name]}}

    hyperparams = []
    normalized_default = ""

    for sub_annotation in annotation_meta.args:
        try:
            sub_output = convert_hyperparam_to_d3m(name, sub_annotation)
        except ValueError:
            continue
        sub_type = sub_output["init_args"]["_structural_type"]
        sub_name = f"{name}__{sub_type}"
        sub_output["name"] = sub_name
        if (
            not normalized_default
            and default != inspect.Parameter.empty
            and _is_instance(default, sub_annotation)
        ):
            normalized_default = sub_name
            sub_output["init_args"]["default"] = _get_default(default)
        hyperparams.append(sub_output)

    output["hyperparams"] = hyperparams

    if normalized_default:
        default = normalized_default

    return output, default


def _process_type_var(name, annotation_meta):
    type_name = annotation_meta.args[0]

    if type_name in {"ArrayLike"}:
        return {
            "name": name,
            "type": "Hyperparameter",
            "init_args": {"_structural_type": "ndarray", "semantic_types": [name]},
        }
    elif type_name == "EstimatorType":
        return {
            "name": name,
            "type": "Hyperparameter",
            "init_args": {"_structural_type": "Estimator", "semantic_types": [name]},
        }

    raise ValueError(f"Unsupported Typevar {name}")


def convert_hyperparam_to_d3m(
    name, annotation, description="", default=inspect.Parameter.empty
):
    annotation_meta = unpack_annotation(annotation)

    if annotation_meta.class_name in {
        "bool",
        "int",
        "float",
        "str",
        "list",
        "dict",
        "Callable",
        "tuple",
    }:
        output = _process_builtins(
            name,
            annotation_meta=annotation_meta,
        )
    elif annotation_meta.class_name == "None":
        output = _process_constant(name, annotation_meta=annotation_meta)
    elif annotation_meta.class_name == "ndarray":
        output = {
            "name": name,
            "type": "Hyperparameter",
            "init_args": {"_structural_type": "ndarray", "semantic_types": [name]},
        }
    elif annotation_meta.class_name == "Union":
        output, default = _process_union(
            name, annotation_meta=annotation_meta, default=default
        )
    elif annotation_meta.class_name == "TypeVar":
        output = _process_type_var(name, annotation_meta=annotation_meta)
    elif annotation_meta.class_name == "Literal":
        output = _process_literal(
            name,
            annotation_meta=annotation_meta,
        )
    else:
        raise ValueError(
            f"Unsupported class_name: {annotation_meta.class_name} in {name}"
        )

    if default != inspect.Parameter.empty:
        output["init_args"]["default"] = _get_default(default)
    if description:
        output["init_args"]["description"] = description

    return output


def convert_attribute_to_d3m(name, annotation, description=""):
    annotation_meta = unpack_annotation(annotation)

    output = {"name": name, "description": description}

    if annotation_meta.class_name in {
        "bool",
        "int",
        "float",
        "str",
        "list",
        "dict",
        "object",
        "tuple",
    }:
        output["type"] = annotation_meta.class_name
    elif annotation == np.ndarray:
        output["type"] = "ndarray"
    elif annotation_meta.class_name == "None":
        output["type"] = "None"
    elif annotation_meta.class_name in {"Union", "List"}:
        # Get the next representation of Union
        output["type"] = str(annotation).replace("typing.", "")
    elif annotation == EstimatorType:
        output["type"] = "sklearn.base.BaseEstimator"
    else:
        raise ValueError(
            f"Unsupported class_name: {annotation_meta.class_name} in {name}"
        )

    return output
