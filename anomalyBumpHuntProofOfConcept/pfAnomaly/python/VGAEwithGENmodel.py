#Currently failing
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GENConv
from torch_geometric.nn import VGAE
import math

class vPFAE_GEN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(vPFAE_GEN, self).__init__()
        layer1_out = in_channels-math.floor((in_channels-out_channels)/3)
        layer2_out = in_channels-math.floor(2*(in_channels-out_channels)/3)

        self.conv1 = GENConv(
            in_channels=in_channels,
            out_channels = layer1_out
        )
        self.conv2 = GENConv(
            in_channels = layer1_out,
            out_channels = layer2_out
        )
        self.conv_mu = GCNConv(layer2_out, out_channels)
        self.conv_logstd = GCNConv(layer2_out, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
