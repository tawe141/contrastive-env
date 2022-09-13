from chemical_env import AtomicEnvironment, BenzeneMD17
import torch


def test_atom_env():
    species = torch.LongTensor([1, 1])
    central = 12
    env_pos = torch.FloatTensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    data = AtomicEnvironment(central, species, env_pos).to_pyg()
    assert torch.allclose(data.x, torch.LongTensor([12, 1, 1]))
    assert torch.allclose(data.edge_index, torch.LongTensor([[1, 2], [0, 0]]))
    assert torch.allclose(data.pos, torch.cat([
        torch.Tensor([[0.0, 0.0, 0.0]]),
        env_pos
    ]))


def test_benzene_md17():
    dataset = BenzeneMD17('.')
    assert len(dataset) == 627983*12
    item = dataset[12]
    assert torch.allclose(item.pos[0], torch.zeros(3))

