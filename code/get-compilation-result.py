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


def evaluate_folder(folder_path, output_csv="compilation_report.csv"):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    # Recursively find all Python files
    python_files = sorted(folder.rglob("*.py"))

    if not python_files:
        print(f"No Python files found in: {folder}")
        return

    results = []

    passed = 0
    failed = 0

    print("=" * 80)
    print("Python Compilation Check")
    print("=" * 80)

    for idx, file_path in enumerate(python_files, start=1):

        success, error = check_compilation(file_path)

        if success:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        relative_path = file_path.relative_to(folder)

        results.append({
            "id": idx,
            "file": str(relative_path),
            "status": status,
            "error": error
        })

        print(f"[{idx}/{len(python_files)}] {status:4} | {relative_path}")

        if not success:
            print(f"    Error: {error}")

    total = len(python_files)

    compilation_rate = (
        passed / total * 100
        if total > 0
        else 0.0
    )

    # Write CSV report
    with open(output_csv, "w", newline="", encoding="utf-8") as f:

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

    print()
    print("=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(f"Total programs       : {total}")
    print(f"Compiled successfully: {passed}")
    print(f"Compilation failed   : {failed}")
    print(f"Compilation rate     : {compilation_rate:.2f}%")
    print("=" * 80)

    print(f"\nDetailed report saved to: {output_csv}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Check compilation rate of Python programs."
    )

    parser.add_argument(
        "folder",
        default="../datasets/",
        help="Folder containing Python files"
    )

    parser.add_argument(
        "--output",
        default="compilation_report.csv",
        help="Output CSV file (default: compilation_report.csv)"
    )

    args = parser.parse_args()

    evaluate_folder(
        args.folder,
        args.output
    )