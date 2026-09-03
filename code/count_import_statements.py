#!/usr/bin/env python3

import argparse
import ast
from collections import Counter
from pathlib import Path


def extract_imports(file_path, top_level_only=True):
    """
    Extract imported packages from a Python file.

    Examples:

        import torch
            -> torch

        import numpy as np
            -> numpy

        import os, sys
            -> os
            -> sys

        from jax import numpy as jnp
            -> jax

        from torch.nn import functional
            -> torch       (if top_level_only=True)
            -> torch.nn    (if top_level_only=False)

    Returns:
        list of package names
    """

    packages = []

    try:
        source = file_path.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(
            source,
            filename=str(file_path)
        )

    except Exception as e:
        print(
            f"[WARNING] Cannot parse {file_path}: {e}"
        )
        return packages

    for node in ast.walk(tree):

        # Example:
        #
        # import torch
        # import numpy as np
        # import os, sys
        #
        if isinstance(node, ast.Import):

            for alias in node.names:

                package = alias.name

                if top_level_only:
                    package = package.split(".")[0]

                packages.append(package)

        # Example:
        #
        # from jax import numpy
        # from torch.nn import functional
        #
        elif isinstance(node, ast.ImportFrom):

            if node.module is None:
                continue

            package = node.module

            if top_level_only:
                package = package.split(".")[0]

            packages.append(package)

    return packages


def analyze_folder(
    folder_path,
    output_file,
    top_level_only=True
):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {folder}"
        )

    # Recursively collect Python files
    python_files = sorted(
        folder.rglob("*.py")
    )

    print("=" * 80)
    print("IMPORT ANALYSIS")
    print("=" * 80)
    print(f"Folder       : {folder}")
    print(f"Python files : {len(python_files)}")
    print("=" * 80)

    import_counter = Counter()

    successfully_parsed = 0
    failed_to_parse = 0

    for idx, file_path in enumerate(
        python_files,
        start=1
    ):

        try:
            source = file_path.read_text(
                encoding="utf-8"
            )

            tree = ast.parse(
                source,
                filename=str(file_path)
            )

        except Exception as e:

            failed_to_parse += 1

            print(
                f"[{idx}/{len(python_files)}] "
                f"FAIL | {file_path}"
            )

            print(
                f"    {type(e).__name__}: {e}"
            )

            continue

        successfully_parsed += 1

        packages = []

        for node in ast.walk(tree):

            # ----------------------------------------------
            # import xxx
            # ----------------------------------------------

            if isinstance(node, ast.Import):

                for alias in node.names:

                    package = alias.name

                    if top_level_only:
                        package = package.split(".")[0]

                    packages.append(package)

            # ----------------------------------------------
            # from xxx import yyy
            # ----------------------------------------------

            elif isinstance(node, ast.ImportFrom):

                if node.module is None:
                    continue

                package = node.module

                if top_level_only:
                    package = package.split(".")[0]

                packages.append(package)

        import_counter.update(packages)

        relative_path = file_path.relative_to(folder)

        print(
            f"[{idx}/{len(python_files)}] "
            f"OK   | {relative_path} "
            f"({len(packages)} imports)"
        )

    # -------------------------------------------------------
    # Sort:
    #
    # 1. number of occurrences descending
    # 2. package name alphabetically if counts are equal
    # -------------------------------------------------------

    sorted_imports = sorted(
        import_counter.items(),
        key=lambda x: (-x[1], x[0])
    )

    # -------------------------------------------------------
    # Write output
    # -------------------------------------------------------

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for package, count in sorted_imports:

            f.write(
                f"{package}\t{count}\n"
            )

    # -------------------------------------------------------
    # Final report
    # -------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL REPORT")
    print("=" * 80)

    print(
        f"Total Python files      : {len(python_files)}"
    )

    print(
        f"Successfully parsed     : {successfully_parsed}"
    )

    print(
        f"Failed to parse         : {failed_to_parse}"
    )

    print(
        f"Unique imported packages: {len(import_counter)}"
    )

    print(
        f"Total import occurrences: {sum(import_counter.values())}"
    )

    print()
    print("Most common imports:")

    for package, count in sorted_imports[:20]:

        print(
            f"{package:30} {count}"
        )

    print("=" * 80)

    print(
        f"Output saved to: {output_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Recursively analyze Python files and "
            "count imported packages."
        )
    )

    parser.add_argument(
        "folder",
        help="Folder containing Python files"
    )

    parser.add_argument(
        "--output",
        default="import_frequency.txt",
        help=(
            "Output filename "
            "(default: import_frequency.txt)"
        )
    )

    parser.add_argument(
        "--full-module",
        action="store_true",
        help=(
            "Count complete module names instead of "
            "only top-level packages."
        )
    )

    args = parser.parse_args()

    analyze_folder(
        folder_path=args.folder,
        output_file=args.output,
        top_level_only=not args.full_module
    )