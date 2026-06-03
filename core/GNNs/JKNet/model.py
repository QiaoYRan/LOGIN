import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge


class JKNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, mode='cat'):
        super(JKNet, self).__init__()
        
        self.dropout = dropout
        self.activation = F.relu
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.jump = JumpingKnowledge(mode)

        if mode == 'cat':
            self.lin1 = nn.Linear(num_layers * hidden_channels, hidden_channels)
        else:
            self.lin1 = nn.Linear(hidden_channels, hidden_channels)

        self.lin2 = nn.Linear(hidden_channels, out_channels)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, x, edge_index):
        #x = data.x
        #edge_index = data.train_edge_index if self.training else data.edge_index

        xs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = self.jump(xs)
        x = self.activation(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x
