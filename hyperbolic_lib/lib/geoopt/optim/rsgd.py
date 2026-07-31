import torch.optim.optimizer
from ..tensor import ManifoldParameter, ManifoldTensor
from .mixin import OptimMixin
from .utils import move_parameters_rsgd

from ..manifolds.lorentz import Lorentz
from ..manifolds.lorentz.math import expmap0, logmap0, expmap, project_u, project, parallel_transport0back

__all__ = ["RiemannianSGD"]


class RiemannianSGD(OptimMixin, torch.optim.Optimizer):
    r"""
    Riemannian Stochastic Gradient Descent with the same API as :class:`torch.optim.SGD`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float
        learning rate
    momentum : float (optional)
        momentum factor (default: 0)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    dampening : float (optional)
        dampening for momentum (default: 0)
    nesterov : bool (optional)
        enables Nesterov momentum (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        stabilize=None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

        self.man_params = []

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults, stabilize=stabilize)

    def euclid_step(self, point, lr, weight_decay, momentum, dampening, nesterov):

        if point.grad is None:
            return point
        grad = point.grad
        if weight_decay != 0:
            grad = grad.add(point, alpha=weight_decay)
        if momentum != 0:
            param_state = self.state[point]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        point.add_(grad, alpha=-lr)

        return point

    def hyperbolic_step(self, point, lr, weight_decay, momentum, dampening, nesterov):

        grad = point.grad

        manifold = point.manifold

        if point.grad is None:
            if isinstance(manifold, Lorentz):
                self.man_params_no_grad.append([point, manifold.k.clone()])
            return point

        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

        if isinstance(manifold, Lorentz):
            self.man_params.append([point, manifold.k.clone()])

        state = self.state[point]

        # State initialization
        if len(state) == 0:
            if momentum > 0:
                state["momentum_buffer"] = grad.clone()


        grad.add_(point, alpha=weight_decay)
        grad = manifold.egrad2rgrad(point, grad)

        if momentum > 0:
            momentum_buffer = state["momentum_buffer"]
            momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
            if nesterov:
                grad = grad.add_(momentum_buffer, alpha=momentum)
            else:
                grad = momentum_buffer

            # we have all the things projected
            new_point, new_momentum_buffer = manifold.retr_transp(
                point, -lr * grad, momentum_buffer
            )
            momentum_buffer.copy_(new_momentum_buffer)
            # use copy only for user facing point
            point.copy_(new_point)
        else:
            new_point = manifold.retr(point, -lr * grad)
            point.copy_(new_point)

        return point

    def step(self, closure=None):
        loss = None
        self.man_params = []
        self.man_params_no_grad = []

        if closure is not None:
            loss = closure()

        k_group = None

        with torch.no_grad():
            for group in self.param_groups:

                if group["name"] == "k_group":
                    k_group = group
                    continue

                if "step" not in group:
                    group["step"] = 0

                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                learning_rate = group["lr"]
                group["step"] += 1

                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianSGD does not support sparse gradients, use SparseRiemannianSGD instead"
                        )
                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = torch.clone(grad).detach()
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        point.copy_(self.hyperbolic_step(point, learning_rate, weight_decay, momentum, dampening, nesterov))
                    else:
                        point.copy_(self.euclid_step(point, learning_rate, weight_decay, momentum, dampening, nesterov))

            if k_group is not None:
                if "step" not in k_group:
                    k_group["step"] = 0

                weight_decay = k_group["weight_decay"]
                momentum = k_group["momentum"]
                dampening = k_group["dampening"]
                nesterov = k_group["nesterov"]
                learning_rate = k_group["lr"]
                k_group["step"] += 1

                for point in k_group["params"]:
                    new_point = self.euclid_step(point, learning_rate, weight_decay, momentum, dampening, nesterov)
                    point.copy_(new_point.clamp(0.05, 9))

            move_parameters_rsgd(self.man_params, self.param_groups, self.state)
            self.move_parameters_no_grad()
            #self.stabilize_group(self.param_groups[1])

        return loss

    @torch.no_grad()
    def move_parameters_no_grad(self):
        for point in self.man_params_no_grad:
            if point[1] != point[0].manifold.k:
                if point[0].manifold.k == 0:
                    print("Warning: Manifold")

                point[0].copy_(project(point[0], k=point[1]))

                # tangent = logmap0(point[0], k=point[1], dim=-1)
                # point[0].copy_(point[0].manifold.expmap0(tangent, dim=-1))

                point[0].copy_(point[0]*torch.sqrt(point[0].manifold.k/point[1]))