from md17_data import BenzeneMD17
from torch_geometric.nn import Sequential, global_add_pool, BatchNorm, GraphNorm
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MD17
from egnn_pytorch.egnn_pytorch_geometric import EGNN_Sparse
import pytorch_lightning as pl
from torch.nn import Embedding, Linear, ModuleList
from torch.nn.functional import mse_loss
import torch
import numpy as np
from itertools import permutations
from multiprocessing import cpu_count
from typing import Union

torch.set_default_dtype(torch.double)
torch.set_default_tensor_type(torch.DoubleTensor)


HIDDEN_WIDTH = 128
MAX_SPECIES = 144
BATCH_SIZE = 32
NUM_WORKERS = cpu_count()
ACTIVATION_FN = torch.nn.SiLU


class ConcatLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, pos):
        return torch.cat([pos, x], dim=1)


class DecoupleLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :3], x[:, 3:]


class Potential(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.embed = Embedding(MAX_SPECIES, HIDDEN_WIDTH)
        self.conv1 = EGNN_Sparse(HIDDEN_WIDTH, norm_feats=True, update_coors=False, soft_edge=1, aggr="sum")
        self.conv2 = EGNN_Sparse(HIDDEN_WIDTH, norm_feats=True, update_coors=False, soft_edge=1, aggr="sum")
        # self.conv3 = EGNN_Sparse(HIDDEN_WIDTH, norm_feats=True, update_coors=False, soft_edge=1, aggr="sum")

        # self.lin = Linear(HIDDEN_WIDTH, 1)
        self.lin = ModuleList([Linear(HIDDEN_WIDTH, 1) for _ in range(MAX_SPECIES)])
        # self.model = Sequential('x, edge_index, pos, batch', [
        #     (Embedding(MAX_SPECIES, HIDDEN_WIDTH), 'x -> x'),
        #
        #     (ConcatLayer(), 'x, pos -> x'),  # needed to get the right ordering for features
        #     (EGNN_Sparse(HIDDEN_WIDTH, update_coors=False, soft_edge=1), 'x, edge_index -> x'),
        #     # (DecoupleLayer(), 'x -> pos, x'),
        #     # (GraphNorm(HIDDEN_WIDTH), 'x -> x'),
        #     # (ACTIVATION_FN(inplace=True), 'x -> x'),
        #
        #     # (ConcatLayer(), 'x, pos -> x'),
        #     (EGNN_Sparse(HIDDEN_WIDTH, update_coors=False, soft_edge=1), 'x, edge_index -> x'),
        #     (DecoupleLayer(), 'x -> pos, x'),
        #     # (GraphNorm(HIDDEN_WIDTH), 'x -> x'),
        #     # (ACTIVATION_FN(inplace=True), 'x -> x'),
        #     # (ConcatLayer(), 'x, pos -> x'),
        #
        #     (Linear(HIDDEN_WIDTH, 1), 'x -> x'),
        #     (global_add_pool, 'x, batch -> x')
        # ])

    def forward(self, z: Union[torch.IntTensor, torch.LongTensor, torch.ByteTensor],
                edge_index: torch.LongTensor,
                pos: Union[torch.FloatTensor, torch.DoubleTensor],
                batch: torch.LongTensor):
        x = self.embed(z)
        c_out = self.conv1(torch.hstack([pos, x]), edge_index, batch=batch)
        # pos, x = x[:, :3], x[:, 3:]
        c_out = self.conv2(c_out, edge_index, batch=batch)
        # c_out = self.conv3(c_out, edge_index, batch=batch)
        pos, x = c_out[:, :3], c_out[:, 3:]
        energy_prediction = torch.zeros(batch[-1]+1)

        for s in set(z):
            idx = torch.argwhere(z == s)
            if s == 1:
                energy_prediction.scatter_add_(0, batch[idx].flatten(), self.lin[s](x[idx]).flatten() / 1000)
            elif s == 12:
                energy_prediction.scatter_add_(0, batch[idx].flatten(), self.lin[s](x[idx]).flatten())
        return energy_prediction

    def training_step(self, batch) -> torch.Tensor:
        z = batch.x.int()
        E_predict = self(z, batch.edge_index, batch.pos, batch.batch)
        loss = mse_loss(E_predict, batch.E)
        E_std = self.trainer.train_dataloader.dataset.datasets._E_std
        self.log('loss', loss.item())
        self.log('E loss (kcal/mol)', loss * E_std * E_std)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-7, momentum=0.9)
        return optimizer

    def train_dataloader(self):
        # def pre_transform(data):
        #     data.energy = (data.energy - -146527.27916135496) / 2.3454991804448895  # hard-coded mean and std
        #     # dists = torch.pdist(data.pos)
        #     # idx0, idx1 = torch.triu_indices(len(data.pos), len(data.pos), 1)
        #     # just for fun, all-to-all...
        #     data.edge_index = torch.from_numpy(
        #         np.array(list(permutations(range(len(data.pos)))))
        #     )
        #     return data
        # return DataLoader(MD17('.', 'benzene', pre_transform=pre_transform), batch_size=BATCH_SIZE, shuffle=True)
        return DataLoader(BenzeneMD17('.'), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


model = Potential()

if __name__ == "__main__":
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model)
