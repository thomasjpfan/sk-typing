import argparse
import inspect

import json

from sklearn.utils import all_estimators
from numpydoc.docscrape import ClassDoc

from .. import get_init_annotations
from ._d3m import get_d3m_representation

__all__ = ["get_output_for_module"]


def _get_output_for_estimator(name, estimator):
    annotations = get_init_annotations(estimator)
    init_params = inspect.signature(estimator).parameters

    class_doc = ClassDoc(estimator)
    descriptions = {
        param.name: " ".join(param.desc) for param in class_doc["Parameters"]
    }

    hyperparmas = []

    for param, annotation in annotations.items():
        try:
            hyperparam = get_d3m_representation(
                param,
                annotation,
                description=descriptions[param],
                default=init_params[param].default,
            )
        except ValueError as e:
            raise ValueError(f"Failed parsing {name}: {e}")
        hyperparmas.append(hyperparam)

    estimator_output = {}
    estimator_output["Hyperparams"] = hyperparmas
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
