import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (..., dim)
        norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # RMS
        return (x / norm) * self.scale

     

x = torch.randn(3, 5)  # e.g., (batch_size=3, features=5)
rmsnorm = RMSNorm(dim=5)
out = rmsnorm(x)
print(out.shape)  # should be (3, 5)
assert out.shape == (3, 5), "Output shape mismatch"