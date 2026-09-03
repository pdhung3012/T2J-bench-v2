#!/usr/bin/env bash

# ============================================================
# Python/JAX Execution Validation
#
# Usage:
#
#   ./check_compilation.sh \
#       ORIGINAL_FOLDER \
#       INPUT_FOLDER \
#       OUTPUT_CSV
#
# Example:
#
#   ./check_compilation.sh \
#       ../pytorch_original \
#       ../jax_qwen_instruct \
#       qwen_compilation_report.csv
#
# Status:
#
#   success : file exists and executes without error
#   fail    : file exists but Python returns an error
#   missing : corresponding file does not exist
#
# ============================================================

set -u


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ORIGINAL_FOLDER="${1:-../datasets/original}"
INPUT_FOLDER="${2:-../datasets/generated}"
OUTPUT_CSV="${3:-compilation_report.csv}"

# Maximum execution time for each program
TIMEOUT_SECONDS=30


# Remove trailing slash
ORIGINAL_FOLDER="${ORIGINAL_FOLDER%/}"
INPUT_FOLDER="${INPUT_FOLDER%/}"


# ------------------------------------------------------------
# Validate folders
# ------------------------------------------------------------

if [[ ! -d "$ORIGINAL_FOLDER" ]]; then
    echo "ERROR: Original folder does not exist:"
    echo "$ORIGINAL_FOLDER"
    exit 1
fi


if [[ ! -d "$INPUT_FOLDER" ]]; then
    echo "ERROR: Input folder does not exist:"
    echo "$INPUT_FOLDER"
    exit 1
fi


# ------------------------------------------------------------
# CSV escape
# ------------------------------------------------------------

csv_escape() {

    local value="$1"

    # Replace " with ""
    value="${value//\"/\"\"}"

    printf '"%s"' "$value"
}


# ------------------------------------------------------------
# Initialize CSV
# ------------------------------------------------------------

echo '"id","file","status","exit_code","error"' > "$OUTPUT_CSV"


# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

total=0
success_count=0
fail_count=0
missing_count=0


echo "================================================================================"
echo "PYTHON / JAX EXECUTION CHECK"
echo "================================================================================"

echo "Original folder : $ORIGINAL_FOLDER"
echo "Input folder    : $INPUT_FOLDER"
echo "Output CSV      : $OUTPUT_CSV"
echo "Timeout         : ${TIMEOUT_SECONDS}s"

echo "================================================================================"


# ------------------------------------------------------------
# Traverse reference/original folder
# ------------------------------------------------------------

while IFS= read -r -d '' original_file; do

    total=$((total + 1))


    # --------------------------------------------------------
    # Relative file path
    #
    # original:
    #
    #   original/kernelbench/level2/test.py
    #
    # generated:
    #
    #   generated/kernelbench/level2/test.py
    #
    # --------------------------------------------------------

    relative_path="${original_file#"$ORIGINAL_FOLDER"/}"

    input_file="$INPUT_FOLDER/$relative_path"


    # ========================================================
    # CASE 1: Missing
    # ========================================================

    if [[ ! -f "$input_file" ]]; then

        status="missing"

        exit_code=""

        error="Corresponding file does not exist."

        missing_count=$((missing_count + 1))


        printf "[%5d] %-8s | %s\n" \
            "$total" \
            "MISSING" \
            "$relative_path"


    # ========================================================
    # CASE 2/3: Execute Python/JAX program
    # ========================================================

    else

        # ----------------------------------------------------
        # Actually execute the program.
        #
        # timeout prevents:
        #
        # - infinite loops
        # - extremely long-running programs
        # - hanging JAX compilation
        #
        # stdout and stderr are captured.
        # ----------------------------------------------------

        error_output=$(
            timeout "${TIMEOUT_SECONDS}s" \
                python3 "$input_file" \
                2>&1
        )

        exit_code=$?


        # ====================================================
        # SUCCESS
        # ====================================================

        if [[ $exit_code -eq 0 ]]; then

            status="success"

            error=""

            success_count=$((success_count + 1))


            printf "[%5d] %-8s | %s\n" \
                "$total" \
                "SUCCESS" \
                "$relative_path"


        # ====================================================
        # FAIL
        # ====================================================

        else

            status="fail"

            fail_count=$((fail_count + 1))


            # ------------------------------------------------
            # GNU timeout uses exit code 124 when timeout occurs
            # ------------------------------------------------

            if [[ $exit_code -eq 124 ]]; then

                error="TIMEOUT after ${TIMEOUT_SECONDS} seconds"

            else

                error="$error_output"

            fi


            printf "[%5d] %-8s | %s\n" \
                "$total" \
                "FAIL" \
                "$relative_path"


            # Display error
            if [[ -n "$error" ]]; then

                echo "         Exit code: $exit_code"

                echo "$error" | sed 's/^/         /'

            fi

        fi

    fi


    # --------------------------------------------------------
    # Save individual result to CSV
    # --------------------------------------------------------

    {
        printf '%d,' "$total"

        csv_escape "$relative_path"

        printf ','

        csv_escape "$status"

        printf ','

        csv_escape "$exit_code"

        printf ','

        csv_escape "$error"

        printf '\n'

    } >> "$OUTPUT_CSV"


done < <(

    find "$ORIGINAL_FOLDER" \
        -type f \
        -name "*.py" \
        -print0 |
    sort -z

)


# ------------------------------------------------------------
# Calculate overall statistics
# ------------------------------------------------------------

non_success=$((fail_count + missing_count))


if [[ $total -gt 0 ]]; then

    success_rate=$(
        awk \
            -v success="$success_count" \
            -v total="$total" \
            'BEGIN {
                printf "%.2f",
                (success / total) * 100
            }'
    )

    fail_rate=$(
        awk \
            -v failed="$non_success" \
            -v total="$total" \
            'BEGIN {
                printf "%.2f",
                (failed / total) * 100
            }'
    )

else

    success_rate="0.00"
    fail_rate="0.00"

fi


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

echo
echo "================================================================================"
echo "FINAL REPORT"
echo "================================================================================"

printf "%-32s : %d\n" \
    "Total expected programs" \
    "$total"

printf "%-32s : %d\n" \
    "Success" \
    "$success_count"

printf "%-32s : %d\n" \
    "Fail" \
    "$fail_count"

printf "%-32s : %d\n" \
    "Missing" \
    "$missing_count"

printf "%-32s : %d\n" \
    "Total unsuccessful" \
    "$non_success"

echo "--------------------------------------------------------------------------------"

printf "%-32s : %s%%\n" \
    "Success rate" \
    "$success_rate"

printf "%-32s : %s%%\n" \
    "Failure rate" \
    "$fail_rate"

echo "================================================================================"

echo
echo "CSV report:"
echo "$OUTPUT_CSV"