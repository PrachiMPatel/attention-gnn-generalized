import torch
from typing import Optional

from torch import arange, Tensor
from torch.nn import Linear, Sequential
from torch.nn.functional import dropout
from torch_geometric.nn import global_max_pool, GATv2Conv, global_max_pool, global_mean_pool, DirGNNConv, GINEConv, GATv2Conv, SAGEConv

class GraphClassifier(torch.nn.Module):
    def __init__(self, hidden_channel_dimensions:list, num_classes:int, edge_dim:Optional[int]=None, pooling_layer:callable=global_mean_pool):
        super().__init__()
        self.layers = Sequential()
        for i in range(1,len(hidden_channel_dimensions)):
            self.layers.append(
                GATv2Conv(in_channels=hidden_channel_dimensions[i-1], out_channels=hidden_channel_dimensions[i], edge_dim=edge_dim)
            )
        self.node_pooling = pooling_layer
        self.linear = Linear(hidden_channel_dimensions[-1], num_classes)
        
    def forward(
        self,
        node_features:Tensor,
        edge_index:Tensor,
        batch:Tensor,
        edge_attr:Optional[Tensor]=None,
        node_subset_indices:Optional[Tensor]=None,
        dropout_percentage:float=0.75
    ):
        node_subset_indices = node_subset_indices if node_subset_indices!=None else arange(node_features.shape[0])
        # print(node_subset_indices)
        x = node_features
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr).relu()
        
        # 2. Readout layer
        # x = self.node_pooling(x[node_subset_indices,:], batch[node_subset_indices])
        x=self.node_pooling(x,batch)

        # 3. Apply a final classifier
        x = dropout(x, p=dropout_percentage, training=self.training)
        x = self.linear(x)
        
        return x

class GraphClassifierv2(torch.nn.Module):
    def __init__(self, hidden_channel_dimensions:list, num_classes:int, edge_dim:Optional[int]=None, pooling_layer:callable=global_mean_pool):
        super().__init__()
        self.layers = Sequential()
        for i in range(1,len(hidden_channel_dimensions)):
            self.layers.append(
                # GINEConv(in_channels=hidden_channel_dimensions[i-1], out_channels=hidden_channel_dimensions[i], edge_dim=edge_dim)
                DirGNNConv(
                    conv=GATv2Conv(in_channels=hidden_channel_dimensions[i-1], out_channels=hidden_channel_dimensions[i])
                    # conv=SAGEConv(in_channels=hidden_channel_dimensions[i-1], out_channels=hidden_channel_dimensions[i])
                )
            )
        self.node_pooling = pooling_layer
        self.linear = Linear(hidden_channel_dimensions[-1], num_classes)
        
    def forward(
        self,
        node_features:Tensor,
        edge_index:Tensor,
        batch:Tensor,
        edge_attr:Optional[Tensor]=None,
        node_subset_indices:Optional[Tensor]=None,
        dropout_percentage:float=0.5
    ):
        node_subset_indices = node_subset_indices if node_subset_indices!=None else arange(node_features.shape[0])
        x = node_features
        for layer in self.layers:
            # x = layer(x, edge_index, edge_attr=edge_attr).relu()
            x = layer(x, edge_index).relu()
        
        # 2. Readout layer
        x = self.node_pooling(x[node_subset_indices,:], batch[node_subset_indices])

        # 3. Apply a final classifier
        x = dropout(x, p=dropout_percentage, training=self.training)
        x = self.linear(x)
        
        return x