import torch
import torch.nn as nn

from .hat_arch import hat_x4
from .pft_arch import pft_x4


class MultiModelAverage(nn.Module):
    """Ensemble that averages two HAT models and a PFT using fixed weights."""

    def __init__(self, hat_weight=0.25, hat_second_weight=0.25, pft_weight=0.25, pft_second_weight=0.25):
        super().__init__()
        weights = torch.tensor([hat_weight, hat_second_weight, pft_weight,pft_second_weight], dtype=torch.float32)
        if not torch.isclose(weights.sum(), torch.tensor(1.0)):
            raise ValueError("The ensemble weights must sum to 1.0.")

        self.register_buffer("ensemble_weights", weights)
        self.hat_1 = hat_x4()
        self.hat_2 = hat_x4()
        self.pft1 = pft_x4()
        self.pft2 = pft_x4()

    def forward(self, x):
        out_hat_primary = self.hat_1(x)
        out_hat_secondary = self.hat_2(x)
        out_pft_primary = self.pft1(x)
        out_pft_secondary = self.pft2(x)

        stacked = torch.stack(
            [out_hat_primary, out_hat_secondary, out_pft_primary, out_pft_secondary], dim=0
        )

        weights = self.ensemble_weights.to(
            dtype=stacked.dtype, device=stacked.device
        ).view(-1, 1, 1, 1, 1)

        return (stacked * weights).sum(dim=0)
