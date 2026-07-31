import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm

from ..manifold import CustomLorentz
from ...geoopt.optim import RiemannianSGD, RiemannianAdam
from ...geoopt.tensor import ManifoldParameter


class KernelLoss(nn.Module):
    def __init__(
            self,
            manifold: CustomLorentz
    ):
        super(KernelLoss, self).__init__()
        self.manifold = manifold

    def forward(self, kernels: torch.Tensor):

        dist_origin = torch.sum(self.manifold.dist0(kernels))

        internal_distance = self.manifold.dist(kernels, kernels.unsqueeze(1))
        internal_distance = torch.triu(internal_distance, diagonal=1)
        internal_distance = internal_distance.clamp(min=1e-1)
        inv_dist_internal = torch.sum(1/internal_distance)

        return 1*dist_origin + inv_dist_internal


def get_learned_kernels(nk: int,
                        dim: int,
                        epochs: int,
                        manifold: CustomLorentz):
    kernels = torch.randn((nk, dim), device="cuda:0")*10
    kernels = ManifoldParameter(manifold.projx(kernels), manifold, requires_grad=True)

    optimizer = RiemannianAdam([kernels], lr=1e-1, weight_decay=1e-5, stabilize=5)

    lr_scheduler = MultiStepLR(optimizer, milestones=[100, 150, 160], gamma=0.1)

    criterion = KernelLoss(manifold).cuda()

    for i in tqdm(range(epochs)):

        loss = criterion(kernels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    torch.cuda.empty_cache()
    print(loss)
    return kernels.data


if __name__ == '__main__':
    get_learned_kernels(10, 100, 100, CustomLorentz())

