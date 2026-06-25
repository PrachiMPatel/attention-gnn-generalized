"""GATv2 graph classifier (verbatim copy of models/gnn/graph_classifier.py).

This module is intentionally a copy rather than an import so the inference
package is self-contained and the platform engineer can lift this folder
out of the repo without needing the surrounding training code.

The trained checkpoint at models/gnn_wits_v1.pt was produced with
hidden_channel_dimensions=[896, 128, 64], num_classes=4, edge_dim=None
(see models/gnn_wits_v1_metadata.json). Do not change those constructor
arguments — they are baked into the saved weights.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import arange, Tensor
from torch.nn import Linear, Sequential
from torch.nn.functional import dropout
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GraphClassifier(torch.nn.Module):
    def __init__(
        self,
        hidden_channel_dimensions: list[int],
        num_classes: int,
        edge_dim: Optional[int] = None,
        pooling_layer: callable = global_mean_pool,
    ):
        super().__init__()
        self.layers = Sequential()
        for i in range(1, len(hidden_channel_dimensions)):
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden_channel_dimensions[i - 1],
                    out_channels=hidden_channel_dimensions[i],
                    edge_dim=edge_dim,
                )
            )
        self.node_pooling = pooling_layer
        self.linear = Linear(hidden_channel_dimensions[-1], num_classes)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_subset_indices: Optional[Tensor] = None,
        dropout_percentage: float = 0.0,
    ):
        node_subset_indices = (
            node_subset_indices
            if node_subset_indices is not None
            else arange(node_features.shape[0])
        )
        x = node_features
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr).relu()
        x = self.node_pooling(x, batch)
        x = dropout(x, p=dropout_percentage, training=self.training)
        x = self.linear(x)
        return x
