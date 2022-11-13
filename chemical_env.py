from typing import List, Union, Tuple
from attr import has
import torch
from torch_geometric.data import Dataset, Data, Batch
import numpy as np
import h5py
from scipy.spatial.distance import cdist
import pdb
import os
from typing import List


def np_to_torch(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).type(torch.get_default_dtype())


class AtomicEnvironment:
    def __init__(self, central_species: int, env_species: torch.LongTensor, env_pos: torch.Tensor):
        self.central_species = central_species
        self.env_species = env_species
        self.env_pos = env_pos

    def to_pyg(self):
        x = torch.cat((torch.LongTensor([self.central_species]), self.env_species), dim=0)
        pos = torch.cat((torch.zeros([1, 3]), self.env_pos), dim=0)
        edge_index = torch.LongTensor([[i+1, 0] for i in range(len(self.env_species))]).transpose(0, 1)
        return Data(x, edge_index, pos=pos)


class EnvBatch(Batch):
    @classmethod
    def from_envs(cls, data_list):
        env_batch = super().from_data_list(data_list)
        first_idx = torch.cumsum(
            torch.LongTensor([len(i.x) for i in data_list]),
            dim=0
        )
        #pdb.set_trace()
        #first_idx -= len(data_list[0].x)
        first_idx = torch.roll(first_idx, 1)
        first_idx[0] = 0
        env_batch.first_idx = first_idx
        return env_batch


class Molecule(Batch):
    total_energy: float
    force: torch.Tensor

    @classmethod
    def from_envs(cls, data_list: List[EnvBatch], total_energy: float, force: torch.FloatTensor):
        mol = super().from_data_list(data_list)
        mol.total_energy = total_energy
        mol.force = force
        return mol


class MolecularBatch:
    def __init__(self, batch_list: List[Batch]):
        self.x = torch.cat([i.x for i in batch_list])
        self.pos = torch.cat([i.pos for i in batch_list])
        self.edge_index = self._collate_edge_index([i.edge_index for i in batch_list])
        self.atom_batch = self._collate_atom_batch([i.batch for i in batch_list])
        self.mol_batch = self._collate_mol_batch([i.num_graphs for i in batch_list])
        self.total_energy = torch.Tensor([i.total_energy for i in batch_list])

    def _collate_edge_index(self, edge_index_list: List[torch.LongTensor]) -> torch.LongTensor:
        graph_edge_count = torch.LongTensor([torch.max(i) for i in edge_index_list]) + 1
        shift = torch.cumsum(graph_edge_count, dim=0) - graph_edge_count[0]
        return torch.cat([
            i + s for i, s in zip(edge_index_list, shift)
        ], dim=1)

    def _collate_atom_batch(self, batch_list: List[torch.Tensor]) -> torch.LongTensor:
        batch_count = torch.LongTensor([torch.max(i) for i in batch_list]) + 1
        shift = torch.cumsum(batch_count, dim=0) - batch_count[0]
        return torch.cat([
            i + s for i, s in zip(batch_list, shift)
        ])

    def _collate_mol_batch(self, num_atom_list: List[int]) -> torch.LongTensor:
        num_atom_list = torch.LongTensor(num_atom_list)
        return torch.cat([torch.LongTensor([i]*num_atom_list[i]) for i in range(len(num_atom_list))])


class BenzeneEnvMD17(Dataset):
    def __init__(self, root, r_cutoff=3.0, transform=None, pre_transform=None, pre_filter=None):
        self.r_cutoff = r_cutoff
        super().__init__(root, transform, pre_transform, pre_filter)
        # To support parallel reading from an hdf5 file, open this under __getitem__
        # see https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983
        # self.f = h5py.File(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'benzene_old_dft.npz'

    def len(self):
        if not hasattr(self, '_length'):
            with np.load(self.raw_paths[0]) as raw:
                self._length = len(raw['E']) * len(raw['z'])
        return self._length

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        basename = os.path.basename(self.raw_file_names)
        return '%s.hdf5' % basename

    def process(self):
        with np.load(self.raw_paths[0], allow_pickle=True) as raw:
            f = h5py.File(self.processed_paths[0], 'w')
            for i in ('E', 'F', 'R', 'z'):
                f.create_dataset(i, data=raw[i])

    def _open_hdf5(self):
        self.f = h5py.File(self.processed_paths[0])

    def cache_in_memory(self):
        if not hasattr(self, 'f'):
            self._open_hdf5()
        tmp = {}
        for key in ('E', 'F', 'R', 'z'):
            tmp[key] = np.array(self.f[key])
        self.f = tmp

    def __getitem__(self, idx):
        if not hasattr(self, 'f'):
            self._open_hdf5()
        return super().__getitem__(idx)

    @staticmethod
    def get_env(z, pos, atom_idx, r_cutoff):
        assert len(z) == len(pos)
        all_dist = cdist(pos[atom_idx].reshape(1, -1), pos).flatten()
        within_cutoff = [i for i in range(len(all_dist)) if all_dist[i] <= r_cutoff and i != atom_idx]
        rel_pos = torch.Tensor(pos[within_cutoff] - pos[atom_idx])
        neighbor_z = torch.LongTensor(z[within_cutoff])
        return AtomicEnvironment(z[atom_idx], env_species=neighbor_z, env_pos=rel_pos).to_pyg()

    def get(self, idx):
        mol_num, atom_num = divmod(idx, len(self.f['z']))
        z = self.f['z']
        pos = self.f['R'][mol_num]
        return self.get_env(z, pos, atom_num, self.r_cutoff)


class BenzeneMD17(BenzeneEnvMD17):
    def __init__(self, root, r_cutoff=3.0, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, r_cutoff, transform, pre_transform, pre_filter)

    def len(self):
        if not hasattr(self, '_length'):
            with np.load(self.raw_paths[0]) as raw:
                self._length = len(raw['E'])
        return self._length

    @property
    def z(self):
        # cache z because it'll always be the same. silly to be doing io for this...
        if not hasattr(self, '_z'):
            self._z = self.f['z']
        return self._z

    def _get_mean(self):
        if not hasattr(self, 'f'):
            self._open_hdf5()
        if not hasattr(self, '_E_mean'):
            self._E_mean = np.mean(self.f['E'])
        return self._E_mean

    def _get_std(self):
        if not hasattr(self, '_E_std'):
            self._E_std = np.std(self.f['E'])
        return self._E_std

    def get(self, mol_num):
        z = self.z
        pos = np_to_torch(self.f['R'][mol_num])
        E = (self.f['E'][mol_num][0] - self._get_mean()) / self._get_std()
        F = np_to_torch(self.f['F'][mol_num])
        data_list = [self.get_env(z, pos, i, self.r_cutoff) for i in range(len(z))]
        return Molecule.from_envs(data_list, total_energy=E, force=F)


def mol_collate(batch):
    return MolecularBatch(batch)


class MoleculeDataLoader(torch.utils.data.DataLoader):
    def __init__(self, 
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs
        ) -> None:

        if 'collate_fn' in kwargs: 
            del kwargs['collate_fn']

        super().__init__(dataset, batch_size, shuffle, collate_fn=mol_collate, **kwargs)

