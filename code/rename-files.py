import os
import re
import shutil
from urllib.parse import urlparse

INPUT_FOLDER = "../datasets/torchleet-prob/"
OUTPUT_FOLDER = "../datasets/torchleet-rename/"
MAPPING_FILE = "../datasets/torchleet-prob/mapping_file_id.txt"


def get_name_from_url(url):
    """Extract output filename from GitHub URL."""

    # Remove markdown/backslash artifacts
    url = url.replace("\\_", "_").rstrip("/")

    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]

    name = parts[-1]

    # If URL points to a notebook, use the parent folder name
    # Example:
    # custom-autograd/custom-autgrad-function_SOLN.ipynb
    # -> custom-autograd.py
    if name.endswith(".ipynb"):
        name = parts[-2]

    # Otherwise URL usually points to a directory
    # benchmark/ -> benchmark.py

    return name + ".py"


def load_mapping(mapping_file):
    mapping = {}

    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("ID"):
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                continue

            file_id, text = parts

            # Extract URL from markdown:
            # [https://...](https://...)
            urls = re.findall(r"https?://[^)\]]+", text)

            if not urls:
                continue

            url = urls[-1]
            mapping[file_id] = get_name_from_url(url)

    return mapping


def rename_files(input_folder, output_folder, mapping):
    os.makedirs(output_folder, exist_ok=True)

    for root, _, files in os.walk(input_folder):
        for filename in files:

            if not filename.endswith(".py"):
                continue

            # h10.py -> h10
            file_id = os.path.splitext(filename)[0]

            if file_id not in mapping:
                print(f"Skipping: {filename}")
                continue

            new_name = mapping[file_id]

            src = os.path.join(root, filename)
            dst = os.path.join(output_folder, new_name)

            shutil.copy2(src, dst)

            print(f"{filename:10} -> {new_name}")


mapping = load_mapping(MAPPING_FILE)

print("Mapping:")
for old, new in mapping.items():
    print(f"{old:5} -> {new}")

print("\nProcessing files...")

rename_files(
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    mapping
)