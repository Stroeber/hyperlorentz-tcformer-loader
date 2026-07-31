import umap
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from lib.lorentz.manifold import CustomLorentz

from embed_utils import set_all_seeds, create_random_points

set_all_seeds(3)

manifold = CustomLorentz(k=1)
x = create_random_points((8, 2), manifold, (-5, 5))

x_logged = manifold.logmap0(x)[..., 1:]

fig, ax = plt.subplots()


ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

ax.scatter(x_logged[..., 0], x_logged[..., 1], c="blue")

ax.set_aspect('equal', adjustable='box')

# Show the plot
plt.show()
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111)

####################################################################################################

manifold = CustomLorentz(k=1)
x = create_random_points((32, 4), manifold, (-5, 5))

x_logged = manifold.logmap0(x)[..., 1:]


hyperbolic_mapper = umap.UMAP(target_metric="euclidean").fit(x_logged)
embedded = hyperbolic_mapper.embedding_
plt.scatter(embedded[:, 0], embedded[:, 1], c="red")

hyperbolic_mapper = umap.UMAP(target_metric="hyperbolic").fit(x)
embedded = hyperbolic_mapper.embedding_
plt.scatter(embedded[:, 0], embedded[:, 1], c="blue")

hyperbolic_mapper = umap.UMAP(target_metric="hyperbolic").fit(manifold.to_poincare(x))
embedded = hyperbolic_mapper.embedding_
plt.scatter(embedded[:, 0], embedded[:, 1], c="pink")

hyperbolic_mapper = umap.UMAP(target_metric="euclidean").fit(x[1:])
embedded = hyperbolic_mapper.embedding_
plt.scatter(embedded[:, 0], embedded[:, 1], c="yellow")

plt.show()
plt.clf()
