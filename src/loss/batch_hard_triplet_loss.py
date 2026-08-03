import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
      super().__init__()
      self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
      labels = labels.view(-1)
      distances = torch.cdist(embeddings, embeddings, p=2)

      same_class = labels.unsqueeze(0).eq(labels.unsqueeze(1))
      identity = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
      positive_mask = same_class & ~identity
      negative_mask = ~same_class

      valid_anchors = positive_mask.any(dim=1) & negative_mask.any(dim=1)
      if not valid_anchors.any():
        return embeddings.sum() * 0.0

      hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
      hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values

      losses = F.relu(hardest_positive - hardest_negative + self.margin)
      return losses[valid_anchors].mean()



