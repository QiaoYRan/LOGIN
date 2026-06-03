import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP as APPNPConv


class APPNP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=10, 
                 dropout=0.5, alpha=0.1):
        super(APPNP, self).__init__()

        self.dropout = dropout
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = APPNPConv(K=num_layers, alpha=alpha)


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, x, edge_index):
        #x = data.x
        #edge_index = data.train_edge_index if self.training else data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x = self.prop(x, edge_index)
        return x
