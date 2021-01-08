import argparse
from inspect import signature

from sklearn.utils import all_estimators


parser = argparse.ArgumentParser(
    description="Generate template for specific sklearn module"
)
parser.add_argument("module", help="module to generate annotations for")

args = parser.parse_args()

estimators = [
    (name, est)
    for name, est in all_estimators()
    if est.__module__.split(".")[1] == args.module
]

imports_set = set()
annotations = []

annotation_template = """
class {class_name}Annotation:
    __estimator__ = {class_name}

    def __init__(
        self,
        {init_params}
    ):
        pass"""

for name, est in estimators:
    est_signature = signature(est)
    params = est_signature.parameters

    param_with_defaults = [f"{name}={p.default}" for name, p in params.items()]
    params_formated = ",\n        ".join([p for p in param_with_defaults])
    module = est.__module__.split(".")[1]

    imports_set.add(f"from sklearn.{module} import {name}")

    annotations.append(
        annotation_template.format(class_name=name, init_params=params_formated)
    )

all_imports = "\n".join(list(imports_set))
all_annotations = "\n\n".join(annotations)

print(f"{all_imports}\n\n{all_annotations}")
