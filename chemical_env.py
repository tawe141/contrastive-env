from typing import List, Union, Tuple
from attr import has
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import h5py
from ase import Atoms
from ase.neighborlist import NeighborList
from scipy.spatial.distance import cdist
import os


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


# class BenzeneMD17(Dataset):
#     def __init__(self, root, r_cutoff=3.0, transform=None, pre_transform=None, pre_filter=None):
#         self.r_cutoff = r_cutoff
#         super().__init__(root, transform, pre_transform, pre_filter)
#
#     @property
#     def raw_file_names(self) -> Union[str, List[str], Tuple]:
#         return 'raw_data/benzene2017_dft.npz'
#
#     @property
#     def processed_file_names(self) -> Union[str, List[str], Tuple]:
#         basename = os.path.basename(self.raw_file_names)
#         return 'processed_data/%s.hdf5' % basename
#
#     def process(self):
#         f = h5py.File(self.processed_file_names, 'w')
#         idx = 0
#         with np.load(self.raw_file_names) as raw:
#             z = raw['Z']
#             R = raw['R']
#             for pos in R:
#                 conformer = Atoms(numbers=z, positions=pos)
#                 for i, p in enumerate(pos):

class BenzeneMD17(Dataset):
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

    def get(self, idx):
        mol_num, atom_num = divmod(idx, len(self.f['z']))
        z = self.f['z']
        pos = self.f['R'][mol_num]
        all_dist = cdist(pos[atom_num].reshape(1, -1), pos).flatten()
        within_cutoff = [i for i in range(len(all_dist)) if all_dist[i] <= self.r_cutoff and i != atom_num]
        rel_pos = torch.Tensor(pos[within_cutoff] - pos[atom_num])
        neighbor_z = torch.LongTensor(z[within_cutoff])
        return AtomicEnvironment(z[atom_num], env_species=neighbor_z, env_pos=rel_pos).to_pyg()

        # conformer = Atoms(numbers=self.f['Z'], positions=self.f['R'][mol_num])

        # nl = NeighborList([self.r_cutoff for _ in range(len(self.f['Z']))], self_interaction=False, bothways=True)
        # nl.update(conformer)
        # i, _ = nl.get_neighbors(atom_num)

