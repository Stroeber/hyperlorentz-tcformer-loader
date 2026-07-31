import torch
from torch.nn.utils.parametrizations import orthogonal
import matplotlib.pyplot as plt
import numpy as np
import umap

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers.linear_layers.FF_betas import LorentzBoost, LorentzRotation_Up, LorentzTransform

from embed_utils import set_all_seeds, create_random_points

set_all_seeds(78)


visualizations = [["original", "black"],
                  # ["euclidean", "green"],
                  ["rotated", "red"],
                  # ["rotated_euclid", "purple"],
                  ["boosted", "blue"],
                  # ["boosted_euclid", "purple"],
                  ]
included = [v[0] for v in visualizations]

manifold = CustomLorentz(k=1)
rotation_layer = LorentzRotation_Up(manifold, 128, 128)
torch.nn.init.orthogonal_(rotation_layer.weight, gain=10)
rotation_layer = orthogonal(rotation_layer, "weight", orthogonal_map="cayley")

boost_layer = LorentzBoost(manifold)
boost_layer_2 = LorentzTransform(manifold, 128, regularize=False)

x = create_random_points((8, 127), manifold, (-5, 5))

rotated_x = rotation_layer(x)
boosted_x = boost_layer(x)
boosted_x_2 = boost_layer_2(rotated_x)

euclid_x = manifold.logmap0(x)[..., 1:].detach().cpu().numpy()

hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid',
                              random_state=42).fit(euclid_x)

x = hyperbolic_mapper.embedding_[:, 0]
y = hyperbolic_mapper.embedding_[:, 1]
z = np.sqrt(1 + np.sum(hyperbolic_mapper.embedding_**2, axis=1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c="blue", cmap='Spectral')
ax.view_init(35, 80)

plt.show()

disk_x = x / (1 + z)
disk_y = y / (1 + z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(disk_x, disk_y, c="blue", cmap='Spectral')

boundary = plt.Circle((0, 0), 1, fc='none', ec='k')
ax.add_patch(boundary)

ax.set_ylim(-1, 1)
ax.set_xlim(-1,  1)
ax.axis("equal")

#ax.axis('off')
plt.show()

#  batchnorm separate boost and rotate
print("break")


