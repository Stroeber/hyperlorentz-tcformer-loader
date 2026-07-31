import torch
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers.BN_betas import LorentzBatchNorm2d_allvar

LOOPS = 1
CHANNELS = 256

manifold = CustomLorentz()
inputs = torch.randn(5, 32, 32, CHANNELS).to("cuda:0")

x = manifold.projx(inputs)
bn = LorentzBatchNorm2d_allvar(manifold, CHANNELS, 0.2).to("cuda:0")


def trace_handler(p):
    output = p.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=20)
    print(output)
    p.export_chrome_trace("./trace_" + str(p.step_num) + ".json")


c = bn(inputs)


print("break")
