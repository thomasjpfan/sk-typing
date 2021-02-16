import re
import sys
from pathlib import Path
import argparse
from sklearn.utils import all_estimators
from numpydoc.docscrape import ClassDoc


parser = argparse.ArgumentParser(description="Add attributes to typing module")
parser.add_argument("module", help="Module to add attributes")

args = parser.parse_args()

estimators = {
    name: est
    for name, est in all_estimators()
    if est.__module__.split(".")[1] == args.module
}

if not estimators:
    print(f"There are no estimators in {args.module}")
    sys.exit(1)


module_file = Path("sk_typing") / f"{args.module}.py"

with module_file.open("r") as f:
    all_lines = f.readlines()


class_re = re.compile(r"class (\w+):")

new_lines = ["from typing import Any\n"]


for line in all_lines:
    new_lines.append(line)

    match = class_re.match(line)
    if not match:
        continue

    name = match.group(1)
    sk_est = estimators[name]
    doc = ClassDoc(sk_est)

    attribute = doc["Attributes"]
    attribute_str = [f"    {p.name}: Any\n" for p in attribute]
    new_lines.extend(attribute_str)
    new_lines.append("\n")

new_file_content = "".join(new_lines)

with module_file.open("w") as f:
    f.write(new_file_content)
