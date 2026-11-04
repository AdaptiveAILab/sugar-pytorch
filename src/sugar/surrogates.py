"""Surrogate-gradient modules for ReLU-like activations (the SUGAR core).

SUGAR layers preserve forward values using the straight-through construction::

    gx = g(x)                                  # defines the backward pass
    fx = f(x)                                  # true activation -> defines the forward pass
    y  = gx - gx.detach() + fx.detach()        # forward == f(x), dy/dx == g'(x)

The forward function ``f`` follows the activation being replaced (the
*target*): ``nn.ReLU`` -> ``torch.nn.functional.relu`` and ``nn.ReLU6`` ->
``torch.nn.functional.relu6``.

* :class:`ReluSurrogate` / :class:`Surrogate` are the generic modules for
  ``nn.ReLU`` targets.
* Both target families support the same built-in variants. ReLU6-target
  layers additionally clamp the gradient to zero for ``x >= 6``.
"""

from typing import Callable, Dict, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SugarLayer",
    "Surrogate",
    "ReluSurrogate",
    "ActivationWrapper",
    "SUGAR_NeLU",
    "SUGAR_BSiLU",
    "SUGAR_LeakyReLU",
    "nelu",
    "noise_relu",
    "bsilu",
    "register_variant",
]


class SugarLayer(nn.Module):
    """Marker base class for every SUGAR replacement module.

    Lets tooling (``sugar.list_activations``, ``sugar.sugar_layer_filter``)
    recognise SUGAR layers without hard-coding each class.
    """


class Surrogate(SugarLayer):
    """Generic FGI module: forward follows ``f``, gradient follows ``g``.

    Parameters
    ----------
    f:
        Callable returning the true activation value (defines the forward pass,
        e.g. ``torch.nn.functional.relu`` or ``torch.nn.functional.relu6``).
    g:
        Callable whose derivative is used as the backward pass.
    *g_args:
        Extra positional arguments forwarded to ``g``.
    """

    def __init__(self, f: Callable, g: Callable, *g_args):
        super().__init__()
        self.f = f
        self.g = g
        self.g_args = g_args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.g(x, *self.g_args)
        fx = self.f(x)
        return gx - gx.detach() + fx.detach()


class ReluSurrogate(SugarLayer):
    """ReLU-forward surrogate module for ``nn.ReLU`` targets.

    ``forward`` returns exactly ``torch.nn.functional.relu(x)``; the gradient
    is ``g'(x)``.
    """

    def __init__(self, g: Callable, *args):
        super().__init__()
        self.g = g
        self.args = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.g(x, *self.args)
        fx = F.relu(x)
        return gx - gx.detach() + fx.detach()


class Relu6Surrogate(SugarLayer):
    """ReLU6 forward values and a surrogate derivative, zero for x >= 6."""

    def __init__(self, g: Callable, *args):
        super().__init__()
        self.g = g
        self.args = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.g(x, *self.args)
        gx = torch.where(x >= 6, torch.zeros_like(gx), gx)
        return gx - gx.detach() + F.relu6(x).detach()


class ActivationWrapper(SugarLayer):
    """Parameterless module applying ``fn`` directly.

    Used for non-FGI baseline runs so that baseline models keep the same module
    structure (and therefore the same hook attachment sets) as FGI models.
    """

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


# ------------------------------------------------------------------
# ReLU6-target family (MobileNetV2 experiments)
# ------------------------------------------------------------------


class SUGAR_NeLU(SugarLayer):
    """NeLU surrogate for ``nn.ReLU6``: forward is
    ``torch.nn.functional.relu6``. The gradient is the NeLU derivative
    for x < 6 and zero for x >= 6.
    """

    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.where(x > 0, x, -self.alpha * torch.reciprocal(1 + x.square()))
        gx = torch.where(x >= 6, torch.zeros_like(gx), gx)
        fx = F.relu6(x)
        y = gx - gx.detach() + fx.detach()
        return y

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"


class SUGAR_BSiLU(SugarLayer):
    """B-SiLU surrogate for ``nn.ReLU6``.

    Same B-SiLU math as :func:`bsilu` for ``nn.ReLU`` targets,
    with the gradient clamped to 0 for ``x >= 6`` to match ReLU6's saturation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = (x + 1.67) * torch.sigmoid(x) - 0.835
        gx[x >= 6] = 0.
        fx = F.relu6(x)
        return gx - gx.detach() + fx.detach()


class SUGAR_LeakyReLU(SugarLayer):
    """Indirect Leaky ReLU surrogate for ``nn.ReLU6``; zero gradient for x >= 6."""

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.leaky_relu(x, negative_slope=self.alpha)
        gx = torch.where(x >= 6, torch.zeros_like(gx), gx)
        fx = F.relu6(x)
        y = gx - gx.detach() + fx.detach()
        return y

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"


# ------------------------------------------------------------------
# Surrogate functions g
# ------------------------------------------------------------------


def bsilu(x):
    """B-SiLU surrogate function (fixed parameter 1.67).

    Registered for ``nn.ReLU`` targets as ``"bsilu"``. The ``nn.ReLU6`` family uses
    :class:`SUGAR_BSiLU` — the same math with the gradient clamped to 0 for
    ``x >= 6``.
    """
    lower_bound = 1.67
    sig_x = torch.sigmoid(x)
    f_x = x * sig_x - lower_bound * (1 - sig_x)
    c = -lower_bound * (1 - torch.sigmoid(torch.tensor(0.0)))
    return f_x - c


def nelu(x: torch.Tensor, alpha: float = 0.05):
    """NeLU surrogate: x for x > 0, otherwise -alpha / (1 + x**2)."""
    return torch.where(x > 0, x, -alpha * torch.reciprocal(1 + x.square()))



def noise_relu(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Directly inject unit positive slopes and Gaussian non-positive slopes.

    The sampled slopes are detached so autograd uses them directly rather
    than differentiating the random samples. The outer SUGAR layer restores
    the target activation's forward values.
    """
    slopes = torch.where(x > 0, torch.ones_like(x), torch.randn_like(x) * std)
    return x * slopes.detach()


# ------------------------------------------------------------------
# Variant registry: (name, target class) -> builder(alpha)
# ------------------------------------------------------------------


def _fixed_g(g):
    """Builder for ReLU-target variants whose g ignores alpha."""
    def build(alpha=None):
        return ReluSurrogate(g)

    return build


def _kw_g(fn, kwarg, default):
    """Builder for ReLU-target variants whose g takes one keyword argument."""
    def build(alpha=None):
        value = default if alpha is None else alpha
        return ReluSurrogate(lambda x: fn(x, **{kwarg: value}))

    return build


_REGISTRY: Dict[Tuple[str, Type[nn.Module]], Callable[..., nn.Module]] = {
    # nn.ReLU targets
    ("leakyrelu", nn.ReLU): _kw_g(F.leaky_relu, "negative_slope", 0.01),
    ("leaky", nn.ReLU): _kw_g(F.leaky_relu, "negative_slope", 0.01),
    ("elu", nn.ReLU): _fixed_g(F.elu),
    ("celu", nn.ReLU): _fixed_g(F.celu),
    ("gelu", nn.ReLU): _fixed_g(F.gelu),
    ("selu", nn.ReLU): _fixed_g(F.selu),
    ("mish", nn.ReLU): _fixed_g(F.mish),
    ("swish", nn.ReLU): _fixed_g(F.silu),
    ("bsilu", nn.ReLU): _fixed_g(bsilu),
    ("nelu", nn.ReLU): _kw_g(nelu, "alpha", 0.05),
    ("noise_relu", nn.ReLU): _kw_g(noise_relu, "std", 0.01),
    # nn.ReLU6 targets
    ("nelu", nn.ReLU6): lambda alpha=None: SUGAR_NeLU(
        0.05 if alpha is None else alpha
    ),
    ("bsilu", nn.ReLU6): lambda alpha=None: SUGAR_BSiLU(),
    ("leaky", nn.ReLU6): lambda alpha=None: SUGAR_LeakyReLU(
        0.01 if alpha is None else alpha
    ),
}


# Both target families expose the same built-in variants. Keep the existing
# named ReLU6 layers for the three original variants.
def _relu6_builder(relu_builder):
    def build(alpha=None):
        layer = relu_builder(alpha)
        return Relu6Surrogate(layer.g, *layer.args)

    return build


for (_name, _target), _builder in list(_REGISTRY.items()):
    if _target is nn.ReLU:
        _REGISTRY.setdefault((_name, nn.ReLU6), _relu6_builder(_builder))


def register_variant(name: str, target: Type[nn.Module], builder: Callable[..., nn.Module]) -> None:
    """Register a named variant for a target activation class.

    ``builder(alpha)`` must return a fresh ``nn.Module`` each
    time it is called (it is invoked once per replaced slot).
    """
    if not (isinstance(target, type) and issubclass(target, nn.Module)):
        raise TypeError("target must be an nn.Module subclass")
    if not callable(builder):
        raise TypeError("builder must be callable")
    _REGISTRY[(name, target)] = builder


