import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import umap

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers.Kernels import get_learned_kernels


def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

set_all_seeds(1)
manifold = CustomLorentz()
kernels = get_learned_kernels(50, 3, 200, manifold)

euclid_kernels = manifold.logmap0(kernels)[..., 1:]
euclid_kernels = euclid_kernels.cpu().numpy()

embedding = euclid_kernels

# reducer = umap.UMAP()
# embedding = reducer.fit_transform(euclid_kernels)

# Use matplotlib to plot the 2D tensor (as an image)
fig, ax = plt.subplots()
ax.scatter(embedding[:, 0], embedding[:, 1])

# Customize the plot (optional)
plt.title('2D Torch Tensor')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()



