import torch
from torch_geometric.nn import PDNConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GraphNorm
import math

class pfAnomaly(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels, hidden_channels):
        super(pfAnomaly, self).__init__()
        
        layer1_out_channels = in_channels-math.floor( (in_channels-out_channels)/4 )
        layer2_out_channels = in_channels-math.floor( 2*(in_channels-out_channels)/4 )
        layer3_out_channels = in_channels-math.floor( 3*(in_channels-out_channels)/4 )

        self.pre_norm = BatchNorm(in_channels = in_channels)
        
        self.conv1 = PDNConv(in_channels = in_channels,
                             out_channels = layer1_out_channels,
                             edge_dim = edge_channels,
                             hidden_channels = hidden_channels)

        self.norm1 = BatchNorm(in_channels = layer1_out_channels)

        self.conv2 = PDNConv(in_channels = layer1_out_channels,
                               out_channels = layer2_out_channels,
                               edge_dim = edge_channels,
                               hidden_channels = hidden_channels)
        
        self.norm2 = GraphNorm(in_channels = layer2_out_channels)

        self.conv3 = PDNConv(in_channels = layer2_out_channels,
                             out_channels = layer3_out_channels,
                             edge_dim = edge_channels,
                             hidden_channels = hidden_channels)

        self.norm3 = GraphNorm(in_channels = layer3_out_channels)
        
        self.conv4 = PDNConv(in_channels = layer3_out_channels,
                             out_channels = out_channels,
                             edge_dim = edge_channels,
                             hidden_channels = hidden_channels)
        
        self.norm4 = GraphNorm(in_channels = out_channels)

    def forward(self, x, edge_index, edge_attr):

        x = self.pre_norm(x=x)

        x = self.conv1(x=x,
                     edge_index=edge_index,
                     edge_attr=edge_attr).relu()
        x = self.norm1(x=x)

        x = self.conv2(x=x,
                       edge_index=edge_index,
                       edge_attr=edge_attr).relu()
        x = self.norm2(x=x)

        x = self.conv3(x=x,
                       edge_index=edge_index,
                       edge_attr=edge_attr).relu()
        x = self.norm3(x=x)

        x = self.conv4(x=x,
                       edge_index=edge_index,
                       edge_attr=edge_attr).relu()
        x = self.norm4(x=x)

        return x
