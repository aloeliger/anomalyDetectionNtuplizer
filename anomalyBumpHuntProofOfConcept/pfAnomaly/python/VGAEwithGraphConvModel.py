import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import VGAE
import math

class vPFAE_Graph(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(vPFAE_Graph, self).__init__()
        layer1_out = in_channels-math.floor((in_channels-out_channels)/3)
        layer2_out = in_channels-math.floor(2*(in_channels-out_channels)/3)
        #self.conv1 = GCNConv(in_channels, layer1_out)
        #self.conv2 = GCNConv(layer1_out, layer2_out)
        self.conv1 = GraphConv(in_channels, layer1_out)
        self.conv2 = GraphConv(layer1_out, layer2_out)
        self.conv_mu = GCNConv(layer2_out, out_channels)
        self.conv_logstd = GCNConv(layer2_out, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
