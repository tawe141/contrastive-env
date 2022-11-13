from torch_geometric.data import Dataset, InMemoryDataset, Data
from typing import Union, List, Tuple
import numpy as np
import torch


class BenzeneMD17(InMemoryDataset):
    def __init__(self, root, r_cutoff=5.0, transform=None, pre_transform=None, pre_filter=None):
        self.r_cutoff = r_cutoff
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self._E_mean = torch.mean(self.data['E'])
        self._E_std = torch.std(self.data['E'])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['benzene2017_dft.npz']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['benzene2017_dft.pt']

    def process(self):
        data = np.load(self.raw_paths[0])
        pos = torch.from_numpy(data['R'])
        z = torch.from_numpy(data['z'])
        E = torch.from_numpy(data['E'])
        E = (E - torch.mean(E)) / torch.std(E)
        F = torch.from_numpy(data['F'])
        idx0, idx1 = torch.triu_indices(pos.shape[1], pos.shape[1], offset=1)
        full_edge_index = torch.vstack([idx0, idx1])

        within_cutoff = torch.cdist(pos, pos)[:, idx0, idx1] <= self.r_cutoff

        data_list = []
        for i in range(len(data['E'])):
            edge_index = full_edge_index[:, within_cutoff[i]]
            edge_index = torch.hstack(
                [
                    edge_index,
                    torch.vstack([edge_index[1], edge_index[0]])
                ]
            )
            data_list.append(
                Data(
                    x=z,
                    edge_index=edge_index,
                    pos=pos[i],
                    E=E[i],
                    F=F[i]
                )
            )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

