from train import get_first_in_batch
import torch


def test_get_first_in_batch():
    x = torch.rand(20)
    batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    result = get_first_in_batch(x, batch)
    assert torch.allclose(result, torch.Tensor([x[0], x[10]]))
