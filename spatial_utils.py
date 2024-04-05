import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Helper function for 2+d distance
def newDistance(a, b):
    return torch.norm(a - b, dim=-1)


# Helper function for edge weights
def makeEdgeWeight(x, edge_index):
    to = edge_index[0]
    fro = edge_index[1]
    # x is the array of node features
    distances = newDistance(x[to], x[fro])

    max_val = torch.max(distances)
    min_val = torch.min(distances)
    rng = max_val - min_val

    edge_weight = (max_val - distances) / rng
    
    return edge_weight

class MaskedMAELoss(nn.Module):
    def __init__(self):
        super(MaskedMAELoss, self).__init__()

    def forward(self, pred, target):
        mask_value = -float('inf')
        mask = target != mask_value
        masked_target = target[mask]

        if masked_target.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        L1_loss = nn.L1Loss()(pred, masked_target)
        return L1_loss
