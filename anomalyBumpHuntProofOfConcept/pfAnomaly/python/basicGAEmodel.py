import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE
import math

class PFAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PFAE, self).__init__()
        #self.conv1 = GCNConv(in_channels, 2*out_channels)
        #self.conv2 = GCNConv(2*out_channels, out_channels)
        layer1_out = in_channels-math.floor((in_channels-out_channels)/3)
        layer2_out = in_channels-math.floor(2*(in_channels-out_channels)/3)
        self.conv1 = GCNConv(in_channels, layer1_out)
        self.conv2 = GCNConv(layer1_out, layer2_out)
        self.conv3 = GCNConv(layer2_out, out_channels)

    def forward(self, x, edge_index):
        #x = self.conv1(x, edge_index).relu()
        #return self.conv2(x, edge_index)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)
