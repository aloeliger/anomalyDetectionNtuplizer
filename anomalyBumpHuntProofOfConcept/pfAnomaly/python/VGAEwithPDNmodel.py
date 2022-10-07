import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import PDNConv
from torch_geometric.nn import VGAE
#from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GraphNorm
#from torch_geometric.nn import LayerNorm
from torch_geometric.nn import TAGConv
import math

class vPFAE_PDN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_features, hidden_channels):
        super(vPFAE_PDN, self).__init__()
        #layer1_out = in_channels-math.floor((in_channels-out_channels)/3)
        #layer2_out = in_channels-math.floor(2*(in_channels-out_channels)/3)
        layer1_out = in_channels-math.floor((in_channels-out_channels)/10)
        layer2_out = in_channels-math.floor(2*(in_channels-out_channels)/10)
        layer3_out = in_channels-math.floor(3*(in_channels-out_channels)/10)
        layer4_out = in_channels-math.floor(4*(in_channels-out_channels)/10)
        layer5_out = in_channels-math.floor(5*(in_channels-out_channels)/10)
        layer6_out = in_channels-math.floor(6*(in_channels-out_channels)/10)
        layer7_out = in_channels-math.floor(7*(in_channels-out_channels)/10)
        layer8_out = in_channels-math.floor(8*(in_channels-out_channels)/10)
        layer9_out = in_channels-math.floor(9*(in_channels-out_channels)/10)

        
        self.conv1 = PDNConv(in_channels=in_channels, 
                             out_channels=layer1_out, 
                             edge_dim=edge_features,
                             hidden_channels=hidden_channels)
        self.norm1 = GraphNorm(in_channels=layer1_out)
        self.conv2 = PDNConv(in_channels=layer1_out, 
                             out_channels=layer2_out, 
                             edge_dim=edge_features,
                             hidden_channels= hidden_channels)
        self.norm2 = GraphNorm(in_channels=layer2_out)
        self.conv3 = PDNConv(in_channels=layer2_out,
                             out_channels=layer3_out,
                             edge_dim=edge_features,
                             hidden_channels=hidden_channels)
        self.norm3 = GraphNorm(in_channels=layer3_out)
        self.conv4 = PDNConv(in_channels=layer3_out,
                             out_channels=layer4_out,
                             edge_dim=edge_features,
                             hidden_channels=hidden_channels)
        self.norm4 = GraphNorm(in_channels=layer4_out)

        self.conv5 = PDNConv(in_channels=layer4_out,
                             out_channels=layer5_out,
                             edge_dim=edge_features,
                             hidden_channels=hidden_channels)
        self.norm5 = GraphNorm(in_channels=layer5_out)

        self.conv6 = PDNConv(in_channels=layer5_out,
                             out_channels=layer6_out,
                             edge_dim=edge_features,
                             hidden_channels=hidden_channels)
        self.norm6 = GraphNorm(in_channels=layer6_out)

        self.conv7 = PDNConv(in_channels=layer6_out,
                             out_channels=layer7_out,
                             edge_dim=edge_features,
                             hidden_channels=hidden_channels)
        self.norm7 = GraphNorm(in_channels=layer7_out)

        self.conv8 = PDNConv(in_channels=layer7_out,
                             out_channels=layer8_out,
                             edge_dim=edge_features,
                             hidden_channels=hidden_channels)
        self.norm8 = GraphNorm(in_channels=layer8_out)

        self.conv9 = PDNConv(in_channels=layer8_out,
                             out_channels=layer9_out,
                             edge_dim=edge_features,
                             hidden_channels=hidden_channels)
        self.norm9 = GraphNorm(in_channels=layer9_out)

        self.conv_mu = PDNConv(in_channels=layer9_out,
                               out_channels=out_channels,
                               edge_dim=edge_features,
                               hidden_channels=hidden_channels)
        self.conv_logstd = PDNConv(in_channels=layer9_out,
                               out_channels=out_channels,
                               edge_dim=edge_features,
                               hidden_channels=hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        #x = self.conv1(x=x, edge_index=edge_index).relu()
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.norm1(x=x)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.norm2(x=x)
        x = self.conv3(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.norm3(x=x)
        x = self.conv4(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.norm4(x=x)
        x = self.conv5(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.norm5(x=x)
        x = self.conv6(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.norm6(x=x)
        x = self.conv7(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.norm7(x=x)
        x = self.conv8(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.norm8(x=x)
        x = self.conv9(x=x, edge_index=edge_index, edge_attr=edge_attr).relu()
        x = self.norm9(x=x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        return self.conv_mu(x=x, edge_index=edge_index, edge_attr=edge_attr), self.conv_logstd(x=x, edge_index=edge_index, edge_attr=edge_attr)
