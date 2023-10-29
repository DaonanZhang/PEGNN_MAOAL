import math
import torch
import numpy as np


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
