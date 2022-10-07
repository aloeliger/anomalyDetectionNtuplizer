import torch
import torch.nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ECConv
from torch_geometric.nn import VGAE
import math

class vPFAE_EC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_features):
        super(vPFAE_EC, self).__init__()
        layer1_out = in_channels-math.floor((in_channels-out_channels)/3)
        layer2_out = in_channels-math.floor(2*(in_channels-out_channels)/3)
        
        self.conv1 = ECConv(
            in_channels = in_channels,
            out_channels=layer1_out,
            nn = torch.nn.Sequential(
                #torch.nn.Linear(edge_features, math.floor((in_channels*layer1_out)/2)),
                #torch.nn.ReLU(),
                #torch.nn.Linear(math.floor((in_channels*layer1_out)/2), in_channels*layer1_out),
                torch.nn.Linear(edge_features, in_channels*layer1_out),
            )
        )

        self.conv2 = ECConv(
            in_channels = layer1_out,
            out_channels = layer2_out,
            nn = torch.nn.Sequential(
                #torch.nn.Linear(edge_features, math.floor((layer1_out*layer2_out)/2)),
                #torch.nn.ReLU(),
                #torch.nn.Linear(math.floor((layer1_out*layer2_out)/2), layer1_out*layer2_out),
                torch.nn.Linear(edge_features, layer1_out*layer2_out)
            )
        )
        self.conv_mu = GCNConv(layer2_out, out_channels)
        self.conv_logstd = GCNConv(layer2_out, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

