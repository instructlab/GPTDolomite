import torch
import torch.nn as nn


class ParameterizedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        std=None,
    ) -> None:
        self.std = std
        super().__init__(in_features, out_features, bias, device, dtype)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if self.bias is not None:
                self.bias.zero_()
