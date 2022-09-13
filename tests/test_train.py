from train import get_first_in_batch, ContrastiveRepresentation
from chemical_env import BenzeneMD17
import torch


def test_get_first_in_batch():
    x = torch.rand(20)
    batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    result = get_first_in_batch(x, batch)
    assert torch.allclose(result, torch.Tensor([x[0], x[10]]))


def test_dropout_representation():
    model = ContrastiveRepresentation()
    dataset = BenzeneMD17('.')
    x, edge_index, pos = dataset[0]
    x, edge_index, pos = x[1], edge_index[1], pos[1]
    batch = torch.zeros_like(x, dtype=torch.long)
    z1 = model.encoder(x, edge_index, pos, batch)
    z2 = model.encoder(x, edge_index, pos, batch)

    assert not torch.allclose(z1, z2)
