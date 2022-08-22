from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.loader import DataLoader
import torch
from torch.nn import ReLU, Linear, Embedding, Dropout
from torch.nn.functional import log_softmax
import pytorch_lightning as pl
from chemical_env import BenzeneMD17


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
        self.encoder = Sequential('x, edge_index, pos', [
            (Embedding(136, hidden_width), 'x -> x'),
            # (Dropout(p=0.1), 'x -> x'),
            (NaiveGCNLayer(hidden_width, hidden_width), 'x, edge_index, pos -> x'),
            ReLU(inplace=True),
            # (Dropout(p=0.1), 'x -> x'),
            (NaiveGCNLayer(hidden_width, hidden_width), 'x, edge_index, pos -> x'),
            ReLU(inplace=True),
            (Linear(hidden_width, hidden_width), 'x -> x')
        ])

# below is for a single query environment in a batch, but we should not restrict ourselves to just a 
# single example in a batch! this amounts to true stochastic gradient descent...
# maybe a better way is written after this.
    # def contrastive_loss(self, z_query, z_pos, z_all):
    #     keys = torch.cat([z_pos.unsqueeze(0), z_all[1:]], dim=0)
    #     inner_prod = keys @ z_query
    #     return -log_softmax(inner_prod)[0]

    # def training_step(self, batch):
    #     query = batch[0]
    #     positive_key = query
    #     z_query = self.encoder(query.x, query.edge_index, query.pos)[0]
    #     z_pos = self.encoder(positive_key.x, positive_key.edge_index, positive_key.pos)[0]
    #     z_all = self.encoder(batch.x, batch.edge_index, batch.pos)

    #     loss = self.contrastive_loss(z_query, z_pos, z_all)
    #     self.log('contrastive_loss', loss)
    #     return loss

    def contrastive_loss(self, z1, z2=None):
        if z2 is None:
            z2 = z1
        proj = z1 @ z2.transpose(0, 1)
        return torch.trace(-log_softmax(proj, dim=0)) / len(z1)

    def training_step(self, batch) -> float:
        z = self.encoder(batch.x, batch.edge_index, batch.pos)

        loss = self.contrastive_loss(z)
        self.log('contrastive_loss', loss)
        if self.global_step % 1000 == 0:
            self.log_representations(z, batch.x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def log_representations(self, z, metadata):
        self.logger.experiment.add_embedding(z, global_step=self.global_step)


class CosineContrastiveRepresentation(ContrastiveRepresentation):
    def contrastive_loss(self, z1, z2=None):
        z1 = z1 / torch.norm(z1, dim=1, keepdim=True)
        if z2 is not None:
            z2 = z2 / torch.norm(z2, dim=1, keepdim=True)
        return super().contrastive_loss(z1, z2)


class PosNoise(CosineContrastiveRepresentation):
    def __init__(self, noise=0.01):
        super().__init__()
        self.noise = noise

    def forward(self, batch):
        z1 = self.encoder(batch.x, batch.edge_index, batch.pos)
        z2 = self.encoder(batch.x, batch.edge_index, batch.pos + self.noise * torch.randn_like(batch.pos))

        loss = self.contrastive_loss(z1, z2)
        self.log('contrastive_loss', loss)
        if self.global_step % 1000 == 0:
            self.log_representations(z, batch.x)
        return loss


model = PosNoise()

dataset = BenzeneMD17('.')
dl = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=6)

if __name__ == "__main__":
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
    trainer.fit(model, dl)
