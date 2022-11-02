from torch_geometric.nn import GCNConv, Sequential, global_mean_pool, global_add_pool, BatchNorm, global_max_pool
# from torch_geometric.loader import DataLoader
import torch
from torch.utils.data import DataLoader
from torch.nn import ReLU, Linear, Embedding, Dropout, SiLU
from torch.nn.functional import log_softmax, cross_entropy, mse_loss
import pytorch_lightning as pl
from chemical_env import BenzeneEnvMD17, BenzeneMD17, MoleculeDataLoader, EnvBatch
from pytorch_lightning.callbacks import ModelCheckpoint
from multiprocessing import cpu_count
from copy import deepcopy
from math import sin, cos
import pdb


HIDDEN_WIDTH = 64
MAX_SPECIES = 144
BATCH_SIZE = 128
NUM_WORKERS = cpu_count()
ACTIVATION_FN = torch.nn.SiLU


# def get_first_idx_in_batch(batch):
#     result = torch.zeros(torch.max(batch)+1, dtype=torch.long)
#     ptr = 1
#     for i in range(1, len(batch)):
#         if batch[i] > batch[i-1]:
#             result[ptr] = i
#             ptr += 1
#     return result
#
#
# def get_first_in_batch(x, batch):
#     return x[get_first_idx_in_batch(batch)]


class NaiveGCNLayer(GCNConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels+3, out_channels)  # "Add" aggregation (Step 5).

    def forward(self, x, edge_index, pos):
        return super().forward(
            torch.cat([x, pos], dim=1),
            edge_index
        )


class ContrastiveRepresentation(pl.LightningModule):
    def __init__(self, rotate=True):
        super().__init__()
        self.rotate = rotate

        self._central_species_weights = Linear(HIDDEN_WIDTH, MAX_SPECIES)
        self.encoder = Sequential('x, edge_index, pos, batch', [
            (Embedding(MAX_SPECIES, HIDDEN_WIDTH), 'x -> x'),
            (Dropout(p=0.1), 'x -> x'),
            (NaiveGCNLayer(HIDDEN_WIDTH, HIDDEN_WIDTH), 'x, edge_index, pos -> x'),
            #(BatchNorm(HIDDEN_WIDTH), 'x -> x'),
            ACTIVATION_FN(inplace=True),
            (Dropout(p=0.1), 'x -> x'),
            (NaiveGCNLayer(HIDDEN_WIDTH, HIDDEN_WIDTH), 'x, edge_index, pos -> x'),
            #(BatchNorm(HIDDEN_WIDTH), 'x -> x'),
            ACTIVATION_FN(inplace=True),
            (Dropout(p=0.1), 'x -> x'),
            (NaiveGCNLayer(HIDDEN_WIDTH, HIDDEN_WIDTH), 'x, edge_index, pos -> x'),
            (global_max_pool, 'x, batch -> x'),
            (Linear(HIDDEN_WIDTH, HIDDEN_WIDTH), 'x -> x')
        ])
        self.potential = Sequential('x, batch', [
            (Linear(HIDDEN_WIDTH, HIDDEN_WIDTH), 'x -> x'),
            ACTIVATION_FN(inplace=True),
            (Linear(HIDDEN_WIDTH, 1), 'x -> x'),
            (global_add_pool, 'x, batch -> x')
        ])

    def contrastive_loss(self, z1, z2=None):
        if z2 is None:
            z2 = z1
        proj = z1 @ z2.transpose(0, 1)
        loss = torch.trace(-log_softmax(proj, dim=0)) / len(z1)
        self.log('contrastive_loss', loss)
        return loss
    
    def rotation_contrastive_loss(self, z, batch):
        a, b, c = torch.rand(3) * 2 * torch.pi - torch.pi
        # rotation matrices from wikipedia
        # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        yaw = torch.Tensor([
            [cos(a), -sin(a), 0],
            [sin(a), cos(a), 0],
            [0, 0, 1]
        ])
        pitch = torch.Tensor([
            [cos(b), 0, sin(b)],
            [0, 1, 0],
            [-sin(b), 0, cos(b)]
        ])
        roll = torch.Tensor([
            [1, 0, 0],
            [0, cos(c), -sin(c)],
            [0, sin(c), cos(c)]
        ])
        R = yaw @ pitch @ roll
    
        z2 = self.encoder(batch.x, batch.edge_index, batch.pos @ R.T, batch.batch)

        # lots of different ways to make the loss function here
        # could do elementwise MSE loss
        # or dot product
        # or use this as the contrastive loss
        return self.contrastive_loss(z, z2)
    

    def contrastive_ramp(self):
        linear_ramp = 0.0001 * (self.global_step - 5000)
        weight = min(1, max(0, linear_ramp))
        self.log('contrastive_ramp', weight)
        return weight

    def energy_ramp(self):
        linear_ramp = 0.0001 * (self.global_step - 3000)
        weight = min(1, max(0, linear_ramp))
        self.log('energy_ramp', weight)
        return weight

    def logistic_central_species_loss(self, embedding, x, first_idx):
        """
        Returns a loss based on predicting what the central atom species is
        TODO: a couple ways to implement this, and I don't know which one is better
        one is implementing an MSE loss wrt the embedding layer of the encoder
        the other is just implementing a logistic regression

        implemented here is a basic classifier (ie approach 1)

        :param batch:
        :return:
        """
        masked_x = deepcopy(x)
        masked_x[first_idx] = 0

        z = self._central_species_weights(
            embedding
            # self.encoder(masked_x, batch.edge_index, batch.pos, batch.batch)
        )

        # loss = cross_entropy(z, get_first_in_batch(batch.x, batch.batch))
        loss = cross_entropy(z, x[first_idx])
        self.log('central_species_loss', loss)
        return loss

    def training_step(self, batch) -> float:
        env_batch, energy_batch = batch['env'], batch['energy']

        z = self.encoder(env_batch.x, env_batch.edge_index, env_batch.pos, env_batch.batch)
        loss = self.logistic_central_species_loss(z, env_batch.x, env_batch.first_idx)

        #if self.contrastive_ramp() > 0:
        #    if self.rotate:
        #        loss += self.contrastive_ramp() * self.rotation_contrastive_loss(z, env_batch)
        #    else:
        #        z2 = self.encoder(env_batch.x, env_batch.edge_index, env_batch.pos,
        #                          env_batch.batch)  # get second for dropout noise
        #        loss += self.contrastive_ramp() * self.contrastive_loss(z, z2)

        if self.energy_ramp() > 0:
            en_z = self.encoder(energy_batch.x, energy_batch.edge_index, energy_batch.pos, energy_batch.atom_batch)
            energy_predict = self.potential(en_z, energy_batch.mol_batch)
            #if self.global_step > 5000:
            #    pdb.set_trace()
            energy_loss = mse_loss(energy_predict.squeeze(), energy_batch.total_energy.squeeze())
            loss += energy_loss
            self.log('scaled_energy_mse', energy_loss)
            # self.log('energy_mse', self.trainer.train_dataloader.loaders['energy']._E_std**2 * energy_loss)

        if self.global_step % 1000 == 0:
            self.log_representations(z, env_batch.x[env_batch.first_idx])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        env_dataset = BenzeneEnvMD17('.')
        env_dataset.cache_in_memory()
        env_dl = DataLoader(env_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=EnvBatch.from_envs, num_workers=NUM_WORKERS//2)
        energy_dataset = BenzeneMD17('.')
        energy_dataset.cache_in_memory()
        energy_dl = MoleculeDataLoader(energy_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS//2)

        return {'env': env_dl, 'energy': energy_dl}

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


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', every_n_train_steps=5000)
    trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
    trainer.fit(model)
