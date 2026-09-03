#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def check_compilation(file_path):
    """
    Compile a Python source file using Python's built-in compiler.

    This checks whether the complete source code can be compiled into
    a Python code object without actually executing the program.

    Returns:
        (True, "") if compilation succeeds.
        (False, error_message) otherwise.
    """

    try:
        # Read complete source file
        source_code = file_path.read_text(
            encoding="utf-8"
        )

        # Actually invoke Python's compiler.
        #
        # mode="exec" is appropriate for complete Python programs.
        compile(
            source_code,
            str(file_path),
            "exec"
        )

        return True, ""

    except SyntaxError as e:
        error_message = (
            f"{e.__class__.__name__}: {e.msg}; "
            f"line={e.lineno}; "
            f"offset={e.offset}"
        )

        return False, error_message

    except UnicodeDecodeError as e:
        return False, f"UnicodeDecodeError: {e}"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def evaluate_folder(
    original_folder_path,
    input_folder_path,
    output_csv="compilation_report.csv"
):
    """
    Traverse all Python files in original_folder.

    For every original file:

        original_folder/a/b/example.py

    look for:

        input_folder/a/b/example.py

    Status:
        success -> corresponding file exists and compiles
        fail    -> corresponding file exists but does not compile
        missing -> corresponding file does not exist

    The total number of programs is determined by original_folder.
    Missing programs count as non-compiled programs.
    """

    original_folder = Path(original_folder_path).resolve()
    input_folder = Path(input_folder_path).resolve()
    output_csv = Path(output_csv)

    # ---------------------------------------------------------
    # Validate folders
    # ---------------------------------------------------------

    if not original_folder.exists():
        raise FileNotFoundError(
            f"Original folder does not exist: {original_folder}"
        )

    if not original_folder.is_dir():
        raise NotADirectoryError(
            f"Original folder is not a directory: {original_folder}"
        )

    if not input_folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_folder}"
        )

    if not input_folder.is_dir():
        raise NotADirectoryError(
            f"Input folder is not a directory: {input_folder}"
        )

    # ---------------------------------------------------------
    # Original folder defines all expected programs
    # ---------------------------------------------------------

    original_files = sorted(
        original_folder.rglob("*.py")
    )

    if not original_files:
        print(
            f"No Python files found in original folder: "
            f"{original_folder}"
        )
        return

    total = len(original_files)

    results = []

    success_count = 0
    fail_count = 0
    missing_count = 0

    print("=" * 100)
    print("PYTHON COMPILATION CHECK")
    print("=" * 100)
    print(f"Original folder : {original_folder}")
    print(f"Input folder    : {input_folder}")
    print(f"Expected files  : {total}")
    print("=" * 100)

    # ---------------------------------------------------------
    # Traverse original files
    # ---------------------------------------------------------

    for idx, original_file in enumerate(
        original_files,
        start=1
    ):

        relative_path = original_file.relative_to(
            original_folder
        )

        input_file = input_folder / relative_path

        # =====================================================
        # CASE 1: Missing file
        # =====================================================

        if not input_file.is_file():

            status = "missing"
            error = "Corresponding file does not exist."

            missing_count += 1

            results.append({
                "id": idx,
                "file": str(relative_path),
                "status": status,
                "error": error
            })

            print(
                f"[{idx:4d}/{total}] "
                f"{status.upper():7} | "
                f"{relative_path}"
            )

            continue

        # =====================================================
        # CASE 2/3: File exists -> compile it
        # =====================================================

        success, error = check_compilation(
            input_file
        )

        if success:

            status = "success"
            success_count += 1

        else:

            status = "fail"
            fail_count += 1

        results.append({
            "id": idx,
            "file": str(relative_path),
            "status": status,
            "error": error
        })

        print(
            f"[{idx:4d}/{total}] "
            f"{status.upper():7} | "
            f"{relative_path}"
        )

        if not success:
            print(
                f"             Error: {error}"
            )

    # ---------------------------------------------------------
    # Calculate statistics
    # ---------------------------------------------------------

    non_compiled = (
        fail_count
        + missing_count
    )

    compilation_rate = (
        success_count / total * 100
        if total > 0
        else 0.0
    )

    failure_rate = (
        non_compiled / total * 100
        if total > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_csv.open(
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

    # ---------------------------------------------------------
    # Print final report
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL COMPILATION REPORT")
    print("=" * 100)

    print(
        f"Total expected programs : {total}"
    )

    print(
        f"Success                 : {success_count}"
    )

    print(
        f"Fail                    : {fail_count}"
    )

    print(
        f"Missing                 : {missing_count}"
    )

    print(
        f"Total non-compiled      : {non_compiled}"
    )

    print("-" * 100)

    print(
        f"Compilation rate        : "
        f"{compilation_rate:.2f}%"
    )

    print(
        f"Non-compilation rate    : "
        f"{failure_rate:.2f}%"
    )

    print("=" * 100)

    print(
        f"CSV report saved to: {output_csv}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compile generated Python/JAX programs and calculate "
            "their compilation rate using an original dataset "
            "as the reference."
        )
    )

    parser.add_argument(
        "original_folder",
        nargs="?",
        default="../datasets/original/",
        help=(
            "Reference folder containing the original Python files."
        )
    )

    parser.add_argument(
        "input_folder",
        nargs="?",
        default="../datasets/generated/",
        help=(
            "Folder containing generated/translated Python files."
        )
    )

    parser.add_argument(
        "--output",
        default="compilation_report.csv",
        help=(
            "Output CSV filename "
            "(default: compilation_report.csv)."
        )
    )

    args = parser.parse_args()

    evaluate_folder(
        args.original_folder,
        args.input_folder,
        args.output
    )