from pathlib import Path
import sys
import argparse
from inspect import signature
from inspect import Parameter

from sklearn.utils import all_estimators


parser = argparse.ArgumentParser(
    description="Generate template for specific sklearn module"
)
parser.add_argument("module", help="module to generate annotations for")
parser.add_argument("--force", action="store_true")

args = parser.parse_args()

estimators = [
    (name, est)
    for name, est in all_estimators()
    if est.__module__.split(".")[1] == args.module
]

if not estimators:
    print(f"There are no estimators in {args.module}")
    sys.exit(1)

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


def _format_param(name, p):
    default = p.default
    if isinstance(default, str):
        return f'{name}="{default}"'
    elif default == Parameter.empty:
        return name
    return f"{name}={default}"


for name, est in estimators:
    est_signature = signature(est)
    params = est_signature.parameters

    param_with_defaults = [_format_param(name, p) for name, p in params.items()]
    params_formated = ",\n        ".join([p for p in param_with_defaults])
    module = est.__module__.split(".")[1]

    imports_set.add(f"from sklearn.{module} import {name}")

    annotations.append(
        annotation_template.format(class_name=name, init_params=params_formated)
    )

all_imports = "\n".join(list(imports_set))
all_imports = f"""
from typing import Optional
from typing import Union
from typing import Callable
from typing import Mapping
from typing import Iterable

from ._typing import EstimatorType
from ._typing import Literal
from ._typing import RandomStateType
from ._typing import ArrayLike
from ._typing import CVType
from ._typing import DType

{all_imports}
"""
all_annotations = "\n\n".join(annotations)

output_path = Path("sk_typing") / f"{args.module}.py"

if output_path.exists() and not args.force:
    print(f"{output_path} already exists pass --force to overwrite")
    sys.exit(0)

with output_path.open("w") as f:
    f.write(f"{all_imports}\n\n{all_annotations}")
print(f"{output_path} generated")
