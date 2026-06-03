import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv


class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, hidden_channels=0,
                 dropout=0.5):
        super(SGC, self).__init__()

        self.dropout = dropout
        self.conv = SGConv(in_channels, out_channels, K=num_layers, cached=True)


    def reset_parameters(self):
        self.conv.reset_parameters()


    def forward(self, x, edge_index):
        #x = data.x
        #edge_index = data.train_edge_index if self.training else data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x, edge_index)
        return x
