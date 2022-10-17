from chemical_env import *
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


def test_benzene_env_md17():
    dataset = BenzeneEnvMD17('.')
    assert len(dataset) == 627983*12
    item = dataset[12]
    assert torch.allclose(item.pos[0], torch.zeros(3))


def test_benzene_md17():
    dataset = BenzeneMD17('.')
    assert len(dataset) == 627983
    assert isinstance(dataset[0], Molecule)
    assert hasattr(dataset[0], 'total_energy')
    assert hasattr(dataset[0], 'force')
    assert isinstance(dataset[0].total_energy, float)
    assert isinstance(dataset[0].force, torch.FloatTensor)
    assert isinstance(dataset[0].pos, torch.FloatTensor)
    assert dataset[0].force.shape == (12, 3)

    torch.set_default_dtype(torch.double)
    assert isinstance(dataset[0].force, torch.DoubleTensor)
    assert isinstance(dataset[0].pos, torch.DoubleTensor)


def test_mol_batch():
    dataset = BenzeneMD17('.')
    mol1, mol2 = dataset[0], dataset[1]
    molbatch = MolecularBatch([mol1, mol2])

    assert len(molbatch.x) == (len(mol1.x) + len(mol2.x))
    assert molbatch.edge_index.shape[1] == (mol1.edge_index.shape[1] + mol2.edge_index.shape[1])
    assert molbatch.mol_batch.shape == (24,)
    assert molbatch.atom_batch.shape == (len(molbatch.x),)

    mol_batch_num = [0] * 12
    mol_batch_num.extend([1]*12)
    assert torch.allclose(molbatch.mol_batch, torch.LongTensor(mol_batch_num))


def test_env_batch():
    dataset = BenzeneEnvMD17('.')
    d1, d2 = dataset[0], dataset[6]
    batch = EnvBatch.from_envs([d1, d2])
    assert torch.allclose(batch.first_idx, torch.LongTensor([0, len(d1)]))
