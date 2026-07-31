import matplotlib.pyplot as plt
import numpy as np

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers.linear_layers.FF_betas import CayleyLinear, LorentzTransform, LorentzBoostScale

from embed_utils import set_all_seeds, create_random_points

set_all_seeds(np.random.randint(19999))
#set_all_seeds(1)

markers = [".", ",", "o", "v", "^", "<", ">"]
n_pts = 5

visualizations = [["original", "black"],
                  ["euclidean", "green"],
                  ["rotated", "red"],
                  ["rotated_euclid", "red"],
                  ["boosted", "blue"],
                  ["boosted_euclid", "blue"],
                  ]
included = [v[0] for v in visualizations]

manifold = CustomLorentz(k=1)
#rotation_layer = LorentzRotation_Up(manifold, 3, 3)
#torch.nn.init.orthogonal_(rotation_layer.weight, gain=10)
#rotation_layer = orthogonal(rotation_layer, "weight", orthogonal_map="cayley")
rotation_layer = CayleyLinear(2, 2, bias=False)


#boost_layer = LorentzPureBoost(manifold, dim=3)
boost_layer = LorentzBoostScale(manifold, init_weight=1.2)
boost_layer_2 = LorentzTransform(manifold, 3, regularize=False)

x = create_random_points((n_pts, 2), manifold, (-7, 7), force_normal=4)

rotated_x = rotation_layer(x[...,1:])
rotated_x = manifold.add_time(rotated_x)
boosted_x = boost_layer(x)

euclid_x = manifold.logmap0(x)[..., 1:].detach().cpu().numpy()
boosted_x_euclid = manifold.logmap0(boosted_x)[..., 1:].detach().cpu().numpy()
rotated_x_euclid = manifold.logmap0(rotated_x)[..., 1:].detach().cpu().numpy()

#boosted_x_euclid = boosted_x[..., 1:].detach().cpu().numpy()
#rotated_x_euclid = rotated_x[..., 1:].detach().cpu().numpy()

fig, ax = plt.subplots()
# ax = plt.figure().add_subplot(projection='3d')

if "euclidean" in included:
    for i in range(n_pts):
        ax.scatter(euclid_x[i, 0], euclid_x[i, 1], marker=markers[i], color=visualizations[included.index("euclidean")][1])

if "boosted_euclid" in included:
    for i in range(n_pts):
        ax.scatter(boosted_x_euclid[i, 0], boosted_x_euclid[i, 1], marker=markers[i], color=visualizations[included.index("boosted_euclid")][1])
if "rotated_euclid" in included:
    for i in range(n_pts):
        ax.scatter(rotated_x_euclid[i, 0], rotated_x_euclid[i, 1], marker=markers[i], color=visualizations[included.index("rotated_euclid")][1])


# Customize the plot (optional)
plt.title('2D Torch Tensor')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
ax.set_xlim([-11, 11])
ax.set_ylim([-11, 11])
ax.set_aspect('equal', adjustable='box')
# Show the plot

boundary = plt.Circle((0, 0), 10, fc='none', ec='k')
ax.add_patch(boundary)

for i in range(10):
    boundary = plt.Circle((0, 0), i, fc='none', ec='k', linestyle="--")
    ax.add_patch(boundary)


plt.show()
plt.clf()

fig = plt.figure()

rotated_x = rotated_x.detach().cpu().numpy()
boosted_x = boosted_x.detach().cpu().numpy()

fig, ax = plt.subplots()
# ax = plt.figure().add_subplot(projection='3d')

if "original" in included:
    for i in range(n_pts):
        ax.scatter(x[i, 1], x[i, 2],marker=markers[i], color=visualizations[included.index("original")][1])

if "boosted" in included:
    for i in range(n_pts):
        ax.scatter(boosted_x[i, 1], boosted_x[i, 2], marker=markers[i], color=visualizations[included.index("boosted")][1])

if "rotated" in included:
    for i in range(n_pts):
        ax.scatter(rotated_x[i, 1], rotated_x[i, 2], marker=markers[i], color=visualizations[included.index("rotated")][1])


# Customize the plot (optional)
plt.title('2D Torch Tensor')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
ax.set_xlim([-11, 11])
ax.set_ylim([-11, 11])
ax.set_aspect('equal', adjustable='box')
# Show the plot

boundary = plt.Circle((0, 0), 10, fc='none', ec='k')
ax.add_patch(boundary)

for i in range(10):
    boundary = plt.Circle((0, 0), i, fc='none', ec='k', linestyle="--")
    ax.add_patch(boundary)


plt.show()
plt.clf()

#
# disk_x = x[..., 1] / (1 + x[..., 0])
# disk_y = x[..., 2] / (1 + x[..., 0])
# ax.scatter(disk_x, disk_y, color=visualizations[included.index("original")][1])
#
# disk_x = boosted_x[..., 1] / (1 + boosted_x[..., 0])
# disk_y = boosted_x[..., 2] / (1 + boosted_x[..., 0])
# ax.scatter(disk_x, disk_y, color=visualizations[included.index("rotated")][1])
#
# disk_x = boosted_x_2[..., 1] / (1 + boosted_x_2[..., 0])
# disk_y = boosted_x_2[..., 2] / (1 + boosted_x_2[..., 0])
# ax.scatter(disk_x.detach(), disk_y.detach(), color=visualizations[included.index("boosted")][1])
#
# test = manifold.add_time(test)
# disk_x = test[..., 1] / (1 + test[..., 0])
# disk_y = test[..., 2] / (1 + test[..., 0])
# ax.scatter(disk_x.detach(), disk_y.detach(), color="yellow")

# np.linalg.norm(euclid_x[1] - rotated_x_euclid[1]) - np.linalg.norm(euclid_x[0] - rotated_x_euclid[0])
# np.linalg.norm(euclid_x[1] - boosted_x_euclid[1]) - np.linalg.norm(euclid_x[0] - boosted_x_euclid[0])

# batchnorm separate boost and rotate

print("break")


