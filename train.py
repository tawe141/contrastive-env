from torch_geometric.nn import GCNConv
import torch
from torch.nn import ReLU, Linear, Embedding
from torch_geometric.nn import Sequential
from torch.nn.functional import log_softmax
import pytorch_lightning as pl


class NaiveGCNLayer(GCNConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels+3, out_channels)  # "Add" aggregation (Step 5).

    def forward(self, x, edge_index, pos):
        return super().forward(
            torch.cat([x, pos], dim=1),
            edge_index
        )

hidden_width = 16


class ContrastiveRepresentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Sequential([
            (Embedding(2, hidden_width), 'x -> x'),
            (NaiveGCNLayer(hidden_width, hidden_width), 'x, edge_index, pos -> x'),
            ReLU(inplace=True),
            (NaiveGCNLayer(hidden_width, hidden_width), 'x, edge_index, pos -> x'),
            ReLU(inplace=True),
            Linear(hidden_width, hidden_width)
        ])

    def contrastive_loss(self, z_query, z_pos, z_neg):
        keys = torch.cat([z_pos, z_neg], dim=0)
        inner_prod = torch.dot(z_query, keys)
        return -log_softmax(inner_prod)[0]

    def training_step(self, batch):
        query, positive_key, negative_key = batch
        z_query = self.encoder(query)
        z_pos = self.encoder(positive_key)
        z_neg = self.encoder(negative_key)

        return self.contrastive_loss(z_query, z_pos, z_neg)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


