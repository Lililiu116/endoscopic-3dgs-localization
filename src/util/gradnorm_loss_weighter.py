from functools import cache, partial
from typing import Optional, List, Tuple, Dict, Union, Iterable

import torch
import torch.distributed as dist
from torch.autograd import grad
import torch.nn.functional as F
from torch import nn, tensor, Tensor
from torch.nn import Module, Parameter

from einops import repeat

# 尝试导入 accelerate，如果没有也能跑
try:
    from accelerate import Accelerator  # type: ignore
except ImportError:
    Accelerator = None  # type: ignore


# helper functions

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers

def l1norm(t, dim: int = -1):
    return F.normalize(t, p=1, dim=dim)


# distributed helpers

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def maybe_distributed_mean(t: Tensor):
    if not is_distributed():
        return t

    dist.all_reduce(t)
    t = t / dist.get_world_size()
    return t


# main class

class GradNormLossWeighter(Module):
    def __init__(
        self,
        *,
        num_losses: Optional[int] = None,
        loss_weights: Optional[Union[List[float], Tensor]] = None,
        loss_names: Optional[Tuple[str, ...]] = None,
        learning_rate: float = 1e-4,
        restoring_force_alpha: float = 0.0,
        grad_norm_parameters: Optional[Parameter] = None,
        accelerator: Optional["Accelerator"] = None,  # type: ignore
        frozen: bool = False,
        initial_losses_decay: float = 1.0,
        update_after_step: float = 0.0,
        update_every: float = 1.0,
        loss_mask: Optional[Union[List[bool], Tensor]] = None,
    ):
        super().__init__()
        assert exists(num_losses) or exists(loss_weights)

        if exists(loss_weights):
            if isinstance(loss_weights, list):
                loss_weights = torch.tensor(loss_weights, dtype=torch.float32)

            num_losses = default(num_losses, loss_weights.numel())
        else:
            loss_weights = torch.ones((num_losses,), dtype=torch.float32)  # type: ignore[arg-type]

        assert len(loss_weights) == num_losses  # type: ignore[arg-type]
        assert num_losses is not None and num_losses > 1, "only makes sense if you have multiple losses"
        assert loss_weights.ndim == 1, "loss weights must be 1 dimensional"  # type: ignore[union-attr]

        self.accelerator = accelerator
        self.num_losses = num_losses
        self.frozen = frozen

        self.loss_names = loss_names
        assert not exists(loss_names) or len(loss_names) == num_losses

        assert restoring_force_alpha >= 0.0

        self.alpha = restoring_force_alpha
        self.has_restoring_force = self.alpha > 0

        if exists(grad_norm_parameters):
            assert isinstance(
                grad_norm_parameters, nn.Parameter
            ), "`grad_norm_parameters` needs to be a `Parameter` in which it receives gradients from all losses"

        self._grad_norm_parameters = [grad_norm_parameters]  # hack

        # loss weights, either learned or static

        self.register_buffer("loss_weights", loss_weights)  # type: ignore[arg-type]

        self.learning_rate = learning_rate

        # initial loss
        # if initial loss decay set to less than 1, will EMA smooth the initial loss

        assert 0 <= initial_losses_decay <= 1.0
        self.initial_losses_decay = initial_losses_decay

        self.register_buffer("initial_losses", torch.zeros(num_losses))

        # for renormalizing loss weights at end

        self.register_buffer("init_loss_weights_for_sum", self.loss_weights.clone())

        # for gradient accumulation

        self.register_buffer("loss_weights_grad", torch.zeros_like(loss_weights), persistent=False)  # type: ignore[arg-type]

        # step, for maybe having schedules etc

        self.register_buffer("step", torch.tensor(0.0))

        # mask, for having gradnorm for some losses but not others - True would be to use grad norm. defaults to all True

        if exists(loss_mask):
            if isinstance(loss_mask, list):
                loss_mask = tensor(loss_mask)

        self.register_buffer(
            "loss_mask",
            default(loss_mask, tensor([True] * num_losses)),  # type: ignore[arg-type]
            persistent=False,
        )

        # can update less frequently, to save on compute

        self.update_after_step = update_after_step
        self.update_every = update_every

        self.register_buffer("initted", torch.tensor(False))

    @property
    def grad_norm_parameters(self):
        return self._grad_norm_parameters[0]

    def backward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        losses,
        activations: Optional[Tensor] = None,  # shared representation / parameters
        freeze: bool = False,
        scale: float = 1.0,
        grad_step: bool = True,
        loss_mask: Optional[List[bool]] = None,
        **backward_kwargs,
    ):
        """
        losses:
            - dict[str, Tensor] （如果你传入 dict）
            - list[Tensor] / tuple[Tensor, ...]
            - Tensor (1D)
        """

        if exists(loss_mask):
            loss_mask = tensor(loss_mask)

        loss_mask = default(loss_mask, self.loss_mask)

        # backward functions dependent on whether using hf accelerate or not

        if exists(self.accelerator) and hasattr(self.accelerator, "backward"):
            backward = self.accelerator.backward
        else:
            backward = lambda l, **kwargs: l.backward(**kwargs)

        backward = partial(backward, **backward_kwargs)

        # increment step

        step = self.step.item()

        self.step.add_(int(self.training and grad_step))

        # losses can be passed in as a NamedTuple or dict

        if isinstance(losses, tuple) and hasattr(losses, "_asdict"):
            losses = losses._asdict()

        if isinstance(losses, dict):
            assert exists(self.loss_names)
            input_loss_names = set(losses.keys())
            assert input_loss_names == set(self.loss_names), f"expect losses named {self.loss_names} but received {input_loss_names}"

            losses = [losses[name] for name in self.loss_names]

        # validate that all the losses are a single scalar

        assert all([loss.numel() == 1 for loss in losses])

        # cast losses to tensor form

        if isinstance(losses, (list, tuple)):
            losses = torch.stack(losses)

        # auto move gradnorm module to the device of the losses

        if self.initted.device != losses.device:
            self.to(losses.device)

        assert losses.ndim == 1, "losses must be 1 dimensional"
        assert losses.numel() == self.num_losses, f"you instantiated with {self.num_losses} losses but passed in {losses.numel()} losses"

        total_weighted_loss = (losses * self.loss_weights.detach()).sum()

        backward(total_weighted_loss * scale, **{**backward_kwargs, "retain_graph": not freeze})

        # handle base frozen case, so one can freeze the weights after a certain number of steps,
        # or just to a/b test against learned gradnorm loss weights

        if (
            self.frozen
            or freeze
            or not self.training
            or step < self.update_after_step
            or (step % self.update_every) != 0
            or not loss_mask.any()
        ):
            return total_weighted_loss

        # validate loss mask shape

        assert loss_mask.shape == losses.shape

        mask = loss_mask

        losses = losses[mask]

        # store initial loss

        if self.has_restoring_force:
            if not self.initted.item():
                initial_losses = maybe_distributed_mean(losses)
                self.initial_losses.copy_(initial_losses)
                self.initted.copy_(True)

            elif self.initial_losses_decay < 1.0:
                meaned_losses = maybe_distributed_mean(losses)
                self.initial_losses[mask].lerp_(meaned_losses, 1.0 - self.initial_losses_decay)

        # determine which tensor to get grad norm from

        grad_norm_tensor = default(activations, self.grad_norm_parameters)

        assert exists(grad_norm_tensor), "you need to either set `grad_norm_parameters` on init or `activations` on backwards"

        grad_norm_tensor.requires_grad_()

        # get grad norm with respect to each loss

        grad_norms = []
        loss_weights = self.loss_weights[mask].clone()
        loss_weights = Parameter(loss_weights)

        for weight, loss in zip(loss_weights, losses):
            gradients, = grad(weight * loss, grad_norm_tensor, create_graph=True, retain_graph=True)

            grad_norm_val = gradients.norm(p=2)
            grad_norms.append(grad_norm_val)

        grad_norms = torch.stack(grad_norms)

        # main algorithm for loss balancing

        grad_norm_average = maybe_distributed_mean(grad_norms.mean())

        if self.has_restoring_force:
            loss_ratio = losses.detach() / self.initial_losses[mask]

            relative_training_rate = l1norm(loss_ratio) * self.num_losses

            gradient_target = (grad_norm_average * (relative_training_rate ** -self.alpha)).detach()
        else:
            num_losses = mask.sum().item()
            gradient_target = repeat(grad_norm_average, " -> l", l=num_losses).detach()

        grad_norm_loss = F.l1_loss(grad_norms, gradient_target)

        backward(grad_norm_loss * scale)

        # accumulate gradients

        self.loss_weights_grad[mask] += loss_weights.grad

        if not grad_step:
            return total_weighted_loss

        # manually take a single gradient step

        updated_loss_weights = loss_weights - self.loss_weights_grad[mask] * self.learning_rate

        renormalized_loss_weights = l1norm(updated_loss_weights) * self.init_loss_weights_for_sum[mask].sum()

        self.loss_weights[mask] = renormalized_loss_weights

        self.loss_weights_grad[mask] = 0.0

        return total_weighted_loss
