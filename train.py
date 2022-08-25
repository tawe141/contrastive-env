from torch_geometric.nn import GCNConv, Sequential, global_mean_pool
from torch_geometric.loader import DataLoader
import torch
from torch.nn import ReLU, Linear, Embedding, Dropout
from torch.nn.functional import log_softmax, cross_entropy, relu
import pytorch_lightning as pl
from chemical_env import BenzeneMD17
from math import sin, cos
from copy import deepcopy


HIDDEN_WIDTH = 16
MAX_SPECIES = 144
BATCH_SIZE = 128


def get_first_idx_in_batch(batch):
    result = torch.zeros(torch.max(batch)+1, dtype=torch.long)
    ptr = 1
    for i in range(1, len(batch)):
        if batch[i] > batch[i-1]:
            result[ptr] = i
            ptr += 1
    return result


def get_first_in_batch(x, batch):
    return x[get_first_idx_in_batch(batch)]


class NaiveGCNLayer(GCNConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels+3, out_channels)  # "Add" aggregation (Step 5).

    def forward(self, x, edge_index, pos):
        return super().forward(
            torch.cat([x, pos], dim=1),
            edge_index
        )


class ContrastiveRepresentation(pl.LightningModule):
    def __init__(self):
        self.num_species = 2
        super().__init__()
        self._central_species_weights = Linear(HIDDEN_WIDTH, MAX_SPECIES)
        self.encoder = Sequential('x, edge_index, pos, batch', [
            (Embedding(MAX_SPECIES, HIDDEN_WIDTH), 'x -> x'),
            # (Dropout(p=0.1), 'x -> x'),
            (NaiveGCNLayer(HIDDEN_WIDTH, HIDDEN_WIDTH), 'x, edge_index, pos -> x'),
            ReLU(inplace=True),
            # (Dropout(p=0.1), 'x -> x'),
            (NaiveGCNLayer(HIDDEN_WIDTH, HIDDEN_WIDTH), 'x, edge_index, pos -> x'),
            (global_mean_pool, 'x, batch -> x'),
            (Linear(HIDDEN_WIDTH, HIDDEN_WIDTH), 'x -> x')
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
        loss = torch.trace(-log_softmax(proj, dim=0)) / len(z1)
        self.log('contrastive_loss', loss)
        return loss
    #
    # def rotation_contrastive_loss(self, batch):
    #     a, b, c = torch.rand(3) * 2 * torch.pi - torch.pi
    #     # rotation matrices from wikipedia
    #     # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    #     yaw = torch.Tensor([
    #         [cos(a), -sin(a), 0],
    #         [sin(a), cos(a), 0],
    #         [0, 0, 1]
    #     ])
    #     pitch = torch.Tensor([
    #         [cos(b), 0, sin(b)],
    #         [0, 1, 0],
    #         [-sin(b), 0, cos(b)]
    #     ])
    #     roll = torch.Tensor([
    #         [1, 0, 0],
    #         [0, cos(c), -sin(c)],
    #         [0, sin(c), cos(c)]
    #     ])
    #     R = yaw @ pitch @ roll
    #
    #     z = self.encoder(batch.x, batch.edge_index, batch.pos)
    #

    def central_species_loss(self, batch):
        """
        Returns a loss based on predicting what the central atom species is
        TODO: a couple ways to implement this, and I don't know which one is better
        one is implementing an MSE loss wrt the embedding layer of the encoder
        the other is just implementing a logistic regression

        implemented here is a basic classifier (ie approach 1)

        :param batch:
        :return:
        """
        masked_x = deepcopy(batch.x)
        masked_x[get_first_idx_in_batch(batch.batch)] = 0

        z = self._central_species_weights(
            relu(self.encoder(masked_x, batch.edge_index, batch.pos, batch.batch))
        )

        loss = cross_entropy(z, get_first_in_batch(batch.x, batch.batch))
        self.log('central_species_loss', loss)
        return loss

    def training_step(self, batch) -> float:
        z = self.encoder(batch.x, batch.edge_index, batch.pos, batch.batch)

        loss = self.contrastive_loss(z) + self.central_species_loss(batch)
        if self.global_step % 1000 == 0:
            self.log_representations(z, get_first_in_batch(batch.x, batch.batch))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def log_representations(self, z, metadata):
        self.logger.experiment.add_embedding(z, global_step=self.global_step, metadata=metadata)


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
            self.log_representations(z1, batch.x)
        return loss


model = CosineContrastiveRepresentation()

dataset = BenzeneMD17('.')
dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

if __name__ == "__main__":
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, dl)
