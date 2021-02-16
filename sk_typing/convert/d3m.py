import argparse
import inspect

import json

from sklearn.utils import all_estimators
from numpydoc.docscrape import ClassDoc
import sklearn

from .. import get_metadata
from ._d3m import get_d3m_representation

__all__ = ["get_output_for_module"]


def _get_output_for_estimator(name, estimator):
    metadata = get_metadata(estimator.__name__)
    annotations = metadata["parameters"]
    # attribute_annot = metadata["attribute"]
    init_params = inspect.signature(estimator).parameters

    class_doc = ClassDoc(estimator)
    param_descriptions = {
        param.name: " ".join(param.desc) for param in class_doc["Parameters"]
    }

    hyperparmas = []

    for param, annotation in annotations.items():

        # Remove parameters that is not JSON serializable
        default = init_params[param].default

        if callable(default):
            continue

        try:
            description = param_descriptions[param]
        except KeyError:
            description = ""

        try:
            hyperparam = get_d3m_representation(
                param,
                annotation,
                description=description,
                default=init_params[param].default,
            )
        except ValueError as e:
            raise ValueError(f"Failed parsing {name}: {e}")

        hyperparmas.append(hyperparam)

    estimator_output = {}
    estimator_output["name"] = f"{estimator.__module__}.{name}"
    estimator_output["common_name"] = name
    estimator_output["description"] = " ".join(
        class_doc["Summary"] + class_doc["Extended Summary"]
    )
    estimator_output["sklearn_version"] = sklearn.__version__
    estimator_output["Hyperparams"] = hyperparmas

    # Add attributes
    estimator_output["Params"] = [
        {"name": p.name, "type": p.type, "description": " ".join(p.desc)}
        for p in class_doc["Attributes"]
    ]

    return estimator_output


def get_output_for_module(module):
    estimators = [
        (name, est)
        for name, est in all_estimators()
        if est.__module__.split(".")[1] == module
    ]
    output = {}
    for name, Est in estimators:
        output[name] = _get_output_for_estimator(name, Est)

    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get d3m overlay for module")
    parser.add_argument("module")

    args = parser.parse_args()
    output = get_output_for_module(args.module)
    print(json.dumps(output, indent=2))
