import argparse
import ast
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODELS = {
    "qwen-instruct": "/home/hungphd/git/Qwen2.5-3B-Instruct",
    "qwen": "Qwen/Qwen2.5-Coder-32B",
    "opencode": "m-a-p/OpenCodeInterpreter-DS-33B",
    "gemma": "google/gemma-3-12b-it",
    "nxcode": "NTQAI/Nxcode-CQ-7B-orpo",
    "phi": "microsoft/Phi-4-mini-instruct",
}


# ============================================================
# Load model
# ============================================================

def load_llm(name):
    model_id = MODELS[name]

    print(f"Loading model: {name}")
    print(f"Model path: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    return tokenizer, model


# ============================================================
# Query LLM
# ============================================================

def ask(tokenizer, model, question, max_new_tokens=2048):
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    answer = output[0][inputs.input_ids.shape[1]:]

    return tokenizer.decode(
        answer,
        skip_special_tokens=True
    )


# ============================================================
# Remove markdown code blocks if the LLM still produces them
# ============================================================

def clean_generated_code(text):
    text = text.strip()

    # Handle:
    # ```python
    # code
    # ```
    if text.startswith("```"):
        lines = text.splitlines()

        # Remove opening ```
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        # Remove closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()


# ============================================================
# Basic syntax validation
# ============================================================

def is_valid_python(code):
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


# ============================================================
# Translate one file
# ============================================================

def translate_file(
    tokenizer,
    model,
    input_file,
    output_file,
    max_retries=3,
    max_new_tokens=2048
):
    print("=" * 80)
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")

    try:
        source_code = input_file.read_text(
            encoding="utf-8"
        )
    except Exception as e:
        print(f"[ERROR] Cannot read {input_file}: {e}")
        return False, str(e)

    question = f"""
Translate the following Python code to TensorFlow.

Requirements:
1. Preserve the original behavior and functionality.
2. Replace PyTorch/NumPy operations with appropriate TensorFlow operations when applicable.
3. Use idiomatic TensorFlow.
4. The generated code should be executable Python code.
5. Return ONLY the translated PyTorch/TensorFlow code.
6. Do not include Markdown code fences.
7. Do not include explanations before or after the code.

Original PyTorch code:

{source_code}
""".strip()

    last_error = None

    for attempt in range(1, max_retries + 1):

        print(
            f"Translation attempt "
            f"{attempt}/{max_retries}"
        )

        try:
            answer = ask(
                tokenizer,
                model,
                question,
                max_new_tokens=max_new_tokens
            )

            translated_code = clean_generated_code(answer)

            # Empty generation
            if not translated_code.strip():
                raise ValueError(
                    "LLM returned an empty response."
                )

            # Check whether generated code is syntactically valid
            valid, syntax_error = is_valid_python(
                translated_code
            )

            if not valid:
                raise ValueError(
                    f"Generated code has syntax error: "
                    f"{syntax_error}"
                )

            # Create output directory
            output_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Save translation
            output_file.write_text(
                translated_code + "\n",
                encoding="utf-8"
            )

            print(
                f"[SUCCESS] Translation saved to "
                f"{output_file}"
            )

            return True, None

        except Exception as e:
            last_error = str(e)

            print(
                f"[FAILED] Attempt {attempt}: "
                f"{last_error}"
            )

            # Clear unused CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if attempt < max_retries:
                time.sleep(1)

    print(
        f"[FAILED PERMANENTLY] {input_file}"
    )

    return False, last_error


# ============================================================
# Translate all Python files in a folder
# ============================================================

def translate_folder(
    tokenizer,
    model,
    input_folder,
    output_folder,
    max_retries=3,
    max_new_tokens=2048
):
    input_folder = Path(input_folder).resolve()
    output_folder = Path(output_folder).resolve()

    python_files = sorted(
        input_folder.rglob("*.py")
    )

    total = len(python_files)

    if total == 0:
        print(
            f"No .py files found in {input_folder}"
        )
        return

    print()
    print("=" * 80)
    print("PyTorch -> TensorFlow Translation")
    print("=" * 80)
    print(f"Input folder : {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Files found  : {total}")
    print("=" * 80)

    success_count = 0
    failed_count = 0

    failed_files = []

    for index, input_file in enumerate(
        python_files,
        start=1
    ):
        # Relative path from input folder
        relative_path = input_file.relative_to(
            input_folder
        )

        # Keep identical directory structure
        output_file = (
            output_folder / relative_path
        )

        print()
        print(
            f"[{index}/{total}] "
            f"Translating {relative_path}"
        )

        success, error = translate_file(
            tokenizer=tokenizer,
            model=model,
            input_file=input_file,
            output_file=output_file,
            max_retries=max_retries,
            max_new_tokens=max_new_tokens
        )

        if success:
            success_count += 1
        else:
            failed_count += 1

            failed_files.append(
                {
                    "file": str(relative_path),
                    "error": error
                }
            )

    # ========================================================
    # Final report
    # ========================================================

    success_rate = (
        success_count / total * 100
        if total > 0
        else 0
    )

    print()
    print("=" * 80)
    print("FINAL TRANSLATION REPORT")
    print("=" * 80)
    print(f"Total files           : {total}")
    print(f"Successfully translated: {success_count}")
    print(f"Failed                : {failed_count}")
    print(f"Success rate          : {success_rate:.2f}%")
    print("=" * 80)

    if failed_files:
        print("\nFailed files:")

        for item in failed_files:
            print(
                f"- {item['file']}"
            )
            print(
                f"  Error: {item['error']}"
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Translate PyTorch files to TensorFlow using "
            "a local LLM."
        )
    )

    parser.add_argument(
        "input_folder",
        help="Folder containing Python files"
    )

    parser.add_argument(
        "output_folder",
        help="Folder to store translated TensorFlow files"
    )

    parser.add_argument(
        "--model",
        default="qwen-instruct",
        choices=MODELS.keys(),
        help="LLM to use (default: qwen-instruct)"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help=(
            "Maximum translation attempts per file "
            "(default: 3)"
        )
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help=(
            "Maximum generated tokens "
            "(default: 2048)"
        )
    )

    args = parser.parse_args()

    # Load model only once
    tokenizer, model = load_llm(
        args.model
    )

    translate_folder(
        tokenizer=tokenizer,
        model=model,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        max_retries=args.max_retries,
        max_new_tokens=args.max_new_tokens
    )