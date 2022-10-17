from typing import List, Union, Tuple
from attr import has
import torch
from torch_geometric.data import Dataset, Data, Batch
import numpy as np
import h5py
from scipy.spatial.distance import cdist
import os
from typing import List


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


class MolecularBatch:
    def __init__(self, batch_list: List[Batch]):
        self.x = torch.cat([i.x for i in batch_list])
        self.pos = torch.cat([i.pos for i in batch_list])
        self.edge_index = self._collate_edge_index([i.edge_index for i in batch_list])
        self.atom_batch = self._collate_atom_batch([i.batch for i in batch_list])
        self.mol_batch = self._collate_mol_batch([i.num_graphs for i in batch_list])

    def _collate_edge_index(self, edge_index_list: List[torch.LongTensor]) -> torch.LongTensor:
        graph_edge_count = torch.LongTensor([max(i) for i in edge_index_list])
        shift = torch.cumsum(graph_edge_count) - graph_edge_count[0]
        return torch.cat([
            i + s for i, s in zip(edge_index_list, shift)
        ], dim=1)

    def _collate_atom_batch(self, batch_list: List[torch.Tensor]) -> torch.LongTensor:
        batch_count = torch.LongTensor([max(i) for i in batch_list])
        shift = torch.cumsum(batch_count) - batch_count[0]
        return torch.cat([
            i + s for i, s in zip(batch_list, shift)
        ])

    def _collate_mol_batch(self, num_atom_list: List[int]) -> torch.LongTensor:
        num_atom_list = torch.LongTensor(num_atom_list)
        shift = torch.cumsum(num_atom_list) - num_atom_list[0]
        return num_atom_list + shift


class BenzeneEnvMD17(Dataset):
    def __init__(self, root, r_cutoff=3.0, transform=None, pre_transform=None, pre_filter=None):
        self.r_cutoff = r_cutoff
        super().__init__(root, transform, pre_transform, pre_filter)
        # To support parallel reading from an hdf5 file, open this under __getitem__
        # see https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983
        # self.f = h5py.File(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'benzene2017_dft.npz'

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
        with np.load(self.raw_paths[0]) as raw:
            f = h5py.File(self.processed_paths[0], 'w')
            for i in ('E', 'F', 'R', 'z'):
                f.create_dataset(i, data=raw[i])

    def __getitem__(self, idx):
        if not hasattr(self, 'f'):
            self.f = h5py.File(self.processed_paths[0])
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
    def len(self):
        if not hasattr(self, '_length'):
            with np.load(self.raw_paths[0]) as raw:
                self._length = len(raw['E'])
        return self._length

    def get(self, mol_num):
        z = self.f['z']
        pos = self.f['R'][mol_num]
        data_list = [self.get_env(z, pos, i, self.r_cutoff) for i in range(len(z))]
        return Batch.from_data_list(data_list)


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
