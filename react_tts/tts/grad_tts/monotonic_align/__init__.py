import numpy as np
import torch

from .core import maximum_path_c


def maximum_path(value, mask):
    """Cython maximum monotonic alignment path.

    value: [B, T_x, T_y]
    mask:  [B, T_x, T_y]
    """
    device = value.device
    dtype = value.dtype

    value = value * mask
    value = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value, dtype=np.int32)

    mask_np = mask.data.cpu().numpy()
    t_x_max = mask_np.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask_np.sum(2)[:, 0].astype(np.int32)

    maximum_path_c(path, value, t_x_max, t_y_max)

    return torch.from_numpy(path).to(device=device, dtype=dtype)
