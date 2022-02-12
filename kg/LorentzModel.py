import math

import torch
import torch.nn as nn
from geoopt import ManifoldParameter

from manifolds import Lorentz


def lorentz_linear(x, weight, scale, bias=None):
    x = x @ weight.transpose(-2, -1)
    time = x.narrow(-1, 0, 1).sigmoid() * scale + 1.1
    if bias is not None:
        x = x + bias
    x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
    x_narrow = x_narrow / ((x_narrow * x_narrow).sum(dim=-1, keepdim=True) / (time * time - 1)).sqrt()
    x = torch.cat([time, x_narrow], dim=-1)
    return x

class HyboNet(torch.nn.Module):
    def __init__(self, d, dim, max_scale, max_norm, margin):
        super(HyboNet, self).__init__()
        self.manifold = Lorentz(max_norm=max_norm)

        self.emb_entity = ManifoldParameter(self.manifold.random_normal((len(d.entities), dim), std=1./math.sqrt(dim)), manifold=self.manifold)
        self.relation_bias = nn.Parameter(torch.zeros((len(d.relations), dim)))
        self.relation_transform = nn.Parameter(torch.empty(len(d.relations), dim, dim))
        nn.init.kaiming_uniform_(self.relation_transform)
        self.scale = nn.Parameter(torch.ones(()) * max_scale, requires_grad=False)
        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        h = self.emb_entity[u_idx]
        t = self.emb_entity[v_idx]
        r_bias = self.relation_bias[r_idx]
        r_transform = self.relation_transform[r_idx]

        h = lorentz_linear(h.unsqueeze(1), r_transform, self.scale, r_bias.unsqueeze(1)).squeeze(1)
        neg_dist = (self.margin + 2 * self.manifold.cinner(h.unsqueeze(1), t).squeeze(1))
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]

