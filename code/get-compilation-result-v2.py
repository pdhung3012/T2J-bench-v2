#!/usr/bin/env python3

import argparse
import csv
import py_compile
from pathlib import Path


def check_compilation(file_path):
    """
    Compile a Python file without executing it.

    Returns:
        (True, "") if compilation succeeds.
        (False, error_message) otherwise.
    """
    try:
        py_compile.compile(
            str(file_path),
            doraise=True
        )
        return True, ""

    except py_compile.PyCompileError as e:
        return False, str(e)

    except Exception as e:
        return False, str(e)


def evaluate_folder(
    original_folder_path,
    input_folder_path,
    output_csv="compilation_report.csv"
):
    original_folder = Path(original_folder_path)
    input_folder = Path(input_folder_path)

    if not original_folder.exists():
        raise FileNotFoundError(
            f"Original folder does not exist: {original_folder}"
        )

    if not input_folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_folder}"
        )

    # -------------------------------------------------------
    # The original folder defines the expected set of files
    # -------------------------------------------------------
    original_files = sorted(
        original_folder.rglob("*.py")
    )

    if not original_files:
        print(
            f"No Python files found in original folder: "
            f"{original_folder}"
        )
        return

    results = []

    passed = 0
    failed = 0
    missing = 0
    compilation_failed = 0

    total = len(original_files)

    print("=" * 80)
    print("Python Compilation Check")
    print("=" * 80)

    print(f"Original folder : {original_folder}")
    print(f"Input folder    : {input_folder}")
    print(f"Expected files  : {total}")
    print("=" * 80)

    # -------------------------------------------------------
    # Traverse ORIGINAL folder
    # -------------------------------------------------------
    for idx, original_file in enumerate(
        original_files,
        start=1
    ):
        # Example:
        #
        # original_folder/basic/e1.py
        #
        # relative_path = basic/e1.py
        #
        # expected input file:
        # input_folder/basic/e1.py

        relative_path = original_file.relative_to(
            original_folder
        )

        input_file = input_folder / relative_path

        # ---------------------------------------------------
        # Missing translated/generated file
        # ---------------------------------------------------
        if not input_file.exists():

            status = "MISSING"
            error = "Corresponding file does not exist."

            failed += 1
            missing += 1

            results.append({
                "id": idx,
                "file": str(relative_path),
                "status": status,
                "error": error
            })

            print(
                f"[{idx}/{total}] "
                f"{status:7} | {relative_path}"
            )

            continue

        # ---------------------------------------------------
        # Corresponding file exists -> compilation check
        # ---------------------------------------------------
        success, error = check_compilation(
            input_file
        )

        if success:
            status = "PASS"
            passed += 1

        else:
            status = "FAIL"
            failed += 1
            compilation_failed += 1

        results.append({
            "id": idx,
            "file": str(relative_path),
            "status": status,
            "error": error
        })

        print(
            f"[{idx}/{total}] "
            f"{status:7} | {relative_path}"
        )

        if not success:
            print(
                f"    Error: {error}"
            )

    # -------------------------------------------------------
    # Compilation rate
    #
    # IMPORTANT:
    # denominator = total files in original_folder
    #
    # Missing files are therefore counted as failures.
    # -------------------------------------------------------

    compilation_rate = (
        passed / total * 100
        if total > 0
        else 0.0
    )

    # -------------------------------------------------------
    # Write CSV report
    # -------------------------------------------------------

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "file",
                "status",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # -------------------------------------------------------
    # Final report
    # -------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL REPORT")
    print("=" * 80)

    print(
        f"Total expected programs : {total}"
    )

    print(
        f"Compiled successfully   : {passed}"
    )

    print(
        f"Compilation failed      : {compilation_failed}"
    )

    print(
        f"Missing files           : {missing}"
    )

    print(
        f"Total non-compiled      : {failed}"
    )

    print(
        f"Compilation rate        : "
        f"{compilation_rate:.2f}%"
    )

    print("=" * 80)

    print(
        f"\nDetailed report saved to: "
        f"{output_csv}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Check compilation rate using an original "
            "folder as the reference dataset."
        )
    )

    parser.add_argument(
        "original_folder",
        nargs="?",
        default="../datasets/original/",
        help=(
            "Reference folder containing the original "
            "Python files"
        )
    )

    parser.add_argument(
        "input_folder",
        nargs="?",
        default="../datasets/generated/",
        help=(
            "Folder containing generated/translated "
            "Python files"
        )
    )

    parser.add_argument(
        "--output",
        default="compilation_report.csv",
        help=(
            "Output CSV file "
            "(default: compilation_report.csv)"
        )
    )

    args = parser.parse_args()

    evaluate_folder(
        args.original_folder,
        args.input_folder,
        args.output
    )