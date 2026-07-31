import time

import torch
import numpy as np

from lib.geoopt.manifolds.lorentz import math, math_new, math_edited
from lib.lorentz.manifold import CustomLorentz


x = torch.rand((32, 32, 32, 3)).cuda()
y = torch.rand((32, 32, 32, 3)).cuda()
z = torch.rand((32, 32, 32, 3)).cuda()

k = torch.tensor(1).cuda()
loop_length = 100000

times = []

x_projected = math.project(x, k=k)
y_projected = math.project(y, k=k)
z_projected = math.project(z, k=k)

start = time.time()
for i in range(loop_length):
    logmap_default = math.logmap(x_projected, y_projected, k=k)

torch.cuda.synchronize()
times.append([time.time() - start])

start = time.time()
for i in range(loop_length):
    logmap_new = math_new.logmap(x_projected, y_projected, k=k)

torch.cuda.synchronize()
times.append([time.time()-start])

times.append([0])

assert torch.allclose(logmap_default, logmap_new)

start = time.time()
for i in range(loop_length):
    logmap0_default = math.logmap0(x_projected, k=k)
torch.cuda.synchronize()
times[0].append(time.time() - start)

start = time.time()
for i in range(loop_length):
    logmap0_new = math_new.logmap0(x_projected, k=k)
torch.cuda.synchronize()
times[1].append(time.time() - start)

times[2].append(0)

assert torch.allclose(logmap0_default, logmap0_new)

logmap_z = math.logmap(x_projected, z_projected, k=k)

start = time.time()
for i in range(loop_length):
    pt_default = math.parallel_transport(x_projected, y_projected, logmap_z, k=k)
torch.cuda.synchronize()
times[0].append(time.time() - start)

start = time.time()
for i in range(loop_length):
    pt_new = math_new.parallel_transport(x_projected, y_projected, logmap_z, k=k)
torch.cuda.synchronize()
times[1].append(time.time() - start)

start = time.time()
for i in range(loop_length):
    pt_edited = math_edited._custom_parallel_transport(x_projected, y_projected, logmap_z, k=k)
torch.cuda.synchronize()
times[2].append(time.time() - start)

assert torch.allclose(pt_default, pt_new) and torch.allclose(pt_default, pt_edited, atol=1e-3, rtol=1e-3)

start = time.time()
for i in range(loop_length):
    pt0_default = math.parallel_transport0back(x_projected, logmap_z, k=k)
torch.cuda.synchronize()
times[0].append(time.time() - start)

start = time.time()
for i in range(loop_length):
    pt0_new = math_new.parallel_transport0back(x_projected, logmap_z, k=k)
torch.cuda.synchronize()
times[1].append(time.time() - start)

start = time.time()
for i in range(loop_length):
    pt0_edited = math_edited._custom_parallel_transport0back(x_projected, logmap_z, k=k, dim=-k)
torch.cuda.synchronize()
times[2].append(time.time() - start)

assert torch.allclose(pt0_default, pt0_new) and torch.allclose(pt0_default, pt0_edited, atol=1e-4, rtol=1e-4)

start = time.time()
for i in range(loop_length):
    pt0back = math.parallel_transport0(y_projected, pt0_default, k=k)
torch.cuda.synchronize()
times[0].append(time.time() - start)

start = time.time()
for i in range(loop_length):
    pt0back_new = math_new.parallel_transport0(y_projected, pt0_default, k=k)
torch.cuda.synchronize()
times[1].append(time.time() - start)

start = time.time()
for i in range(loop_length):
    pt0back_edited = math_edited._custom_parallel_transport0(y_projected, pt0_default, k=k, dim=-k)
torch.cuda.synchronize()
times[2].append(time.time() - start)

assert torch.allclose(pt0back, pt0back_new) and torch.allclose(pt0back, pt0back_edited, atol=1e-4, rtol=1e-4)


print(np.array(times))

print("break")

