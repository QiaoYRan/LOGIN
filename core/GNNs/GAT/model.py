import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                use_ln=False, dropout=0.6, activ=F.elu, heads=8):
        super(GAT, self).__init__()

        self.use_ln = use_ln
        self.dropout = dropout
        self.activ = activ

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=self.dropout)
        self.ln1 = nn.LayerNorm(hidden_channels * heads)

        self.conv2 = nn.ModuleList()
        self.ln2 = nn.ModuleList()
        for _ in range(num_layers-2):
            self.conv2.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=self.dropout))
            self.ln2.append(nn.LayerNorm(hidden_channels * heads))

        self.conv2.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=self.dropout))


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.ln1.reset_parameters()
        for conv in self.conv2:
            conv.reset_parameters()
        for ln in self.ln2:
            ln.reset_parameters()


    def forward(self, x, edge_index):
        #x = data.x
        #edge_index = data.train_edge_index if self.training else data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        if self.use_ln:
            x = self.ln1(x)
        x = self.activ(x)

        for conv, ln in zip(self.conv2[:-1], self.ln2):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if self.use_ln:
                x = ln(x)
            x = self.activ(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2[-1](x, edge_index)
        return x
