import random

import numpy as np
import torch

from lib.lorentz.manifold import CustomLorentz


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_random_points(size: tuple or list, manifold: CustomLorentz,
                         dist_range: tuple or list = (0, 1),
                         force_normal: int = -1) -> torch.Tensor:

    x = torch.ones(size).uniform_(dist_range[0], dist_range[1])

    if force_normal != -1:
        x = x * (force_normal / torch.norm(x, dim=-1, keepdim=True))

    return manifold.add_time(x)


def restore_center(manifold: CustomLorentz, original_points: torch.Tensor, transformed_points: torch.Tensor):
    original_center = manifold.centroid(original_points)
    euclid_center = manifold.logmap0(original_center)

    new_center = transformed_points.mean(dim=0)

    return transformed_points - new_center + euclid_center
