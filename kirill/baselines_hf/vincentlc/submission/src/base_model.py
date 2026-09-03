import torch

class BaseDeepFakeModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: torch.Tensor [B, C, H, W] in [0,1]

        Returns:
            torch.Tensor [B] scores in [0,1]
        """
        raise NotImplementedError