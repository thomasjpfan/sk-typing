import json
from pathlib import Path
import argparse

from sk_typing.convert.d3m import get_output_for_module
from sk_typing import _MODULES


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate all d3m modules")
    parser.add_argument(
        "--path", default="artifacts", help="directory to place overlay files"
    )

    args = parser.parse_args()

    output_dir = Path(args.path)
    output_dir.mkdir(exist_ok=True)

    for mod in _MODULES:
        output = get_output_for_module(mod)
        output_path = output_dir / f"{mod}.json"
        print(f"writing sklearn.{mod} to {output_path}")
        with output_path.open("w") as f:
            json.dump(output, f, indent=2)

    print(f"overlay files written to {output_dir} directory")
