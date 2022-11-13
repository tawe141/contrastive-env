from md17_data import BenzeneMD17
import torch


def test_get():
    dataset = BenzeneMD17('.')
    data = dataset[0]
    assert isinstance(data.x, torch.ByteTensor)  # uint8
    assert data.pos.shape == (12, 3)
    assert len(data.x) == 12

