import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(
        "Memory:",
        round(
            torch.cuda.get_device_properties(i).total_memory / 1024**3,
            2
        ),
        "GB"
    )