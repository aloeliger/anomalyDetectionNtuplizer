import torch
import torch.nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINEConv
from torch_geometric.nn import VGAE
import math

class vPFAE_GINE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_features):
        super(vPFAE_GINE, self).__init__()
        layer1_out = in_channels-math.floor((in_channels-out_channels)/3)
        layer2_out = in_channels-math.floor(2*(in_channels-out_channels)/3)
        
        self.conv1 = GINEConv(
            nn = torch.nn.Sequential(
                torch.nn.Linear(in_channels, layer1_out),
                #torch.nn.ReLU(),
                #torch.nn.Linear(in_channels, in_channels),
                #torch.nn.Sigmoid(),
                #torch.nn.Linear(in_channels, layer1_out),
            ),
            edge_dim = edge_features
        )
        self.conv2 = GINEConv(
            nn = torch.nn.Sequential(
                torch.nn.Linear(layer1_out, layer2_out),
                #torch.nn.ReLU(),
                #torch.nn.Linear(layer1_out, layer1_out),
                #torch.nn.Sigmoid(),
                #torch.nn.Linear(layer1_out, layer2_out),
            ),
            edge_dim = edge_features
        )
        self.conv_mu = GCNConv(layer2_out, out_channels)
        self.conv_logstd = GCNConv(layer2_out, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
