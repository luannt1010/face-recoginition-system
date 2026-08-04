import math
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

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # theta < pi-m <-> cos(theta)>cos(pi-m)
        self.threshold = math.cos(math.pi - self.m)
        # msin(m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w_scaled = torch.div(self.W, torch.norm(self.W, p=2, dim=1, keepdim=True).clamp_min(self.eps))
        emb_scaled = torch.div(embeddings, torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp_min(self.eps))

        cosine = torch.mm(emb_scaled, w_scaled.T).clamp(-1+self.eps, 1-self.eps)

        # sin(theta) = sqrt(1 - cos(theta)^2)
        sine_theta = (1.0 - cosine**2).clamp(min=self.eps).sqrt()
        # cos(theta+m)
        cos_theta_m = cosine * self.cos_m - sine_theta * self.sin_m

        # theta in [0, pi] -> want theta+m<=pi so theta<=pi-m <-> cos(theta)>=cos(pi-m) -> cos(theta+m)
        # large angle, low similarity -> valid
        # If cos(theta)<cos(pi-m) <-> theta>pi-m <-> theta+m>pi -> large angle, large similarity ->invalid so cosine-mm help down similarity
        cos_theta_m = torch.where(cosine > self.threshold, cos_theta_m, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # If j != y
        theta_diff = cosine * (1 - one_hot)
        # If j == y
        theta_similar = cos_theta_m * one_hot

        logits = self.s * (theta_diff + theta_similar)
        loss = F.cross_entropy(logits, labels)
        return loss