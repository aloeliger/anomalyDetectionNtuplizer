# File for holding our model

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm, LayerNorm, GraphNorm
from torch_geometric.nn import GAE

class recoAnomaly(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(recoAnomaly, self).__init__()

        layer1_out= in_channels - int((in_channels - out_channels)/2)

        self.pre_norm = BatchNorm(in_channels = in_channels)
        self.conv1 = GCNConv(
            in_channels=in_channels,
            out_channels= layer1_out
        )
        self.act1 = torch.nn.ReLU()

        self.layerNorm = LayerNorm(
            in_channels=layer1_out
        )

        self.conv2 = GCNConv(
            in_channels=layer1_out,
            out_channels=out_channels
        )
        self.act2 = torch.nn.ReLU()

    def forward(self, x, edge_index):
        x = self.pre_norm(x=x)
        x = self.conv1(x=x, edge_index=edge_index)
        x = self.act1(x)

        x = self.layerNorm(x=x)

        x = self.conv2(x=x, edge_index=edge_index)
        x = self.act2(x)

        return x