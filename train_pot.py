# from turtle import forward
import pytorch_lightning as pl
from chemical_env import BenzeneMD17, MoleculeDataLoader
import torch
from train import CosineContrastiveRepresentation
# from torch_geometric.loader import DataLoader

# TODO: infer this directly from the env encoding module
HIDDEN_WIDTH = 16
BATCH_SIZE = 128


env_model = CosineContrastiveRepresentation.load_from_checkpoint('saved_checkpoints/epoch=1-step=85000.ckpt')

class LinearPotential(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.pot = torch.nn.Linear(HIDDEN_WIDTH, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch):
        # with torch.no_grad():
        embeddings = env_model.encoder(batch.x, batch.edge_index, batch.pos, batch.atom_batch)
        per_site_energy = self.pot(embeddings)
        total_energy = torch.zeros(batch.mol_batch[-1]+1).scatter_add_(0, batch.mol_batch, per_site_energy.flatten())
        return torch.nn.functional.mse_loss(total_energy, batch.total_energy)


model = LinearPotential()

dataset = BenzeneMD17('.')
dl = MoleculeDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

if __name__ == "__main__":
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, dl)
