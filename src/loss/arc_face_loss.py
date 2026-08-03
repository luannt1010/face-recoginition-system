import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512, m: float = 0.5, s: float = 64.0):
      super().__init__()
      self.m = m
      self.s = s
      self.eps = 1e-8
      self.W = nn.Parameter(torch.empty(num_classes, embedding_dim))
      nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
      w_scaled = torch.div(self.W, torch.norm(self.W, p=2, dim=1, keepdim=True).clamp_min(self.eps))
      emb_scaled = torch.div(embeddings, torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp_min(self.eps))

      cosine = torch.mm(emb_scaled, w_scaled.T).clamp(-1+self.eps, 1-self.eps)

      theta = cosine.acos()
      one_hot = torch.zeros_like(cosine)
      one_hot.scatter_(1, labels.view(-1, 1), 1.0)

      # If j != y
      theta_diff = cosine * (1 - one_hot)
      # If j == y
      theta_similar = torch.cos(theta + self.m) * one_hot

      logits = self.s * (theta_diff + theta_similar)
      loss = F.cross_entropy(logits, labels)
      return loss