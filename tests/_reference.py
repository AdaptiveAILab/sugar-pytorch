"""Reference copies of the original experiment implementations.

Sources (as they were at migration time):
- experiments/cnn/cnn/sugar.py
- experiments/cnn/cnn/train.py   (local walker + inline baseline Surrogate class)
- experiments/applications/mobilenetv2/mobilenet/sugar.py
- experiments/toy/relu_fgi.py

These are the golden reference for behavioural parity of the `sugar` package.
The public B-SiLU naming is used here. Leaky dispatch intentionally accepts
the requested alpha, matching the corrected package API; other reference math
is retained for parity checks, except noise injection now uses detached
Gaussian slopes as intended rather than zero inactive gradients.

Note: ``MobilenetSUGAR_NeLU`` below keeps the original uncapped ``F.relu``
forward pass although it replaces ``nn.ReLU6``.  That behaviour was later
identified as a bug and fixed to ``F.relu6`` in the package, so no parity test
covers this class (the golden mobilenet nelu entry was re-recorded from the
fixed code).
"""

import torch
import torch.nn as nn
from torch._dynamo.eval_frame import OptimizedModule


# ===========================================================================
# experiments/cnn/cnn/sugar.py
# ===========================================================================

class CnnReluSurrogate(torch.nn.Module):
    def __init__(self, g, *args):
        super(CnnReluSurrogate, self).__init__()
        self.g = g
        self.args = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.g(x, *self.args)
        fx = torch.nn.functional.relu(x)
        return gx - gx.detach() + fx.detach()


def cnn_replace_layers(model, old, new_fn):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            cnn_replace_layers(module, old, new_fn)
        if isinstance(module, old):
            setattr(model, n, new_fn())


def cnn_get_surrogate_layer(g, *args, compile_layer=True):
    holder = {}

    def _factory():
        layer = CnnReluSurrogate(g, *args)
        if compile_layer:
            layer = torch.compile(layer)
            holder["cls"] = OptimizedModule
        else:
            holder["cls"] = CnnReluSurrogate
        return layer

    _factory.layer_type = lambda: holder["cls"]
    return _factory


def cnn_bsilu(x):
    lower_bound = 1.67
    sig_x = torch.sigmoid(x)
    f_x = x * sig_x - lower_bound * (1 - sig_x)
    c = -lower_bound * (1 - torch.sigmoid(torch.tensor(0.0)))
    return f_x - c


def cnn_nelu(x: torch.Tensor, alpha: float = 0.05):
    return torch.where(x > 0, x, -alpha * torch.reciprocal(1 + x.square()))



def cnn_noise_relu(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    slopes = torch.where(x > 0, torch.ones_like(x), torch.randn_like(x) * std)
    return x * slopes.detach()


def cnn_dispatch_g(fgi_type, alpha=0.1):
    """Reference g-selection, with configurable Leaky slope."""
    if fgi_type in ("leakyrelu", "leaky"):
        return lambda x: torch.nn.functional.leaky_relu(x, negative_slope=alpha)
    elif fgi_type == "elu":
        return torch.nn.functional.elu
    elif fgi_type == "celu":
        return torch.nn.functional.celu
    elif fgi_type == "gelu":
        return torch.nn.functional.gelu
    elif fgi_type == "selu":
        return torch.nn.functional.selu
    elif fgi_type == "mish":
        return torch.nn.functional.mish
    elif fgi_type == "swish":
        return torch.nn.functional.silu
    elif fgi_type == "nelu":
        return lambda x: cnn_nelu(x, alpha=alpha)
    elif fgi_type == "bsilu":
        return lambda x: cnn_bsilu(x)
    elif fgi_type == "noise_relu":
        return lambda x: cnn_noise_relu(x, std=alpha)
    raise ValueError(f"Invalid FGI type {fgi_type}")


def cnn_perform_relu_fgi(model, fgi_type="relu", alpha=0.1, compile_layer=True):
    """experiments/cnn/cnn/sugar.py perform_relu_fgi, verbatim."""
    if fgi_type in ("NONE", "relu", "none"):
        return (model, torch.nn.ReLU)
    g = cnn_dispatch_g(fgi_type, alpha)
    factory = cnn_get_surrogate_layer(g, compile_layer=compile_layer)
    cnn_replace_layers(model, torch.nn.ReLU, factory)
    layer_type = factory.layer_type()
    return (model, layer_type)


# ===========================================================================
# experiments/cnn/cnn/train.py (local walker + inline baseline Surrogate)
# ===========================================================================

def cnn_train_replace_layers(module, old_layer_type, new_layer_fn):
    for name, child in module.named_children():
        if isinstance(child, old_layer_type):
            setattr(module, name, new_layer_fn())
        else:
            cnn_train_replace_layers(child, old_layer_type, new_layer_fn)


def make_cnn_train_surrogate(fn):
    """The inline ``class Surrogate(nn.Module)`` from cnn/train.py."""
    class Surrogate(nn.Module):
        def forward(self, x):
            return fn(x)

    return Surrogate


# ===========================================================================
# experiments/applications/mobilenetv2/mobilenet/sugar.py
# ===========================================================================

class MobilenetSUGAR_NeLU(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.where(x > 0, x, -self.alpha * torch.reciprocal(1 + x.square()))
        y = gx - gx.detach() + torch.nn.functional.relu(x).detach()
        return y


class MobilenetSUGAR_BSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = (x + 1.67) * torch.sigmoid(x) - 0.835
        gx[x >= 6] = 0.
        return gx - gx.detach() + torch.nn.functional.relu6(x).detach()


class MobilenetSUGAR_LeakyReLU(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.where((x > 0), 1, self.alpha)
        gx[x >= 6] = 0.
        m = x * gx.detach()
        y = m - m.detach() + torch.nn.functional.relu6(x).detach()
        return y


def mobilenet_replace_modules_(m, old_cls, new_ctor):
    count = 0
    for name, child in list(m.named_children()):
        # First recurse
        count += mobilenet_replace_modules_(child, old_cls, new_ctor)
        # Then possibly replace this child
        if isinstance(child, old_cls):
            setattr(m, name, new_ctor())
            count += 1
    return count


def mobilenet_list_activations(model):
    types = set()
    for mod in model.modules():
        if isinstance(mod, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Hardsigmoid,
                            MobilenetSUGAR_NeLU, MobilenetSUGAR_LeakyReLU,
                            MobilenetSUGAR_BSiLU)):
            types.add(type(mod).__name__)
    return types


# ===========================================================================
# experiments/toy/relu_fgi.py
# ===========================================================================

class ToyReluSurrogate(torch.nn.Module):
    def __init__(self, g, *args):
        super(ToyReluSurrogate, self).__init__()
        self.g = g
        self.args = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.g(x, *self.args)
        fx = torch.nn.functional.relu(x)
        return gx - gx.detach() + fx.detach()


def toy_replace_layers(model, old, new_fn):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            toy_replace_layers(module, old, new_fn)
        if isinstance(module, old):
            setattr(model, n, new_fn())


def toy_fgi_relu(model, g, *args):
    def factory(g=g, args=args):
        return ToyReluSurrogate(g, *args)
    toy_replace_layers(model, torch.nn.ReLU, factory)


def toy_custom_func(x: torch.Tensor):
    fx = (x + 1.67) * torch.sigmoid(x) - 0.835
    return fx


def toy_perform_relu_fgi(model, fgi_type):
    if fgi_type == "NONE":
        layer_type = torch.nn.ReLU
    elif fgi_type == "LeakyReLU":
        toy_fgi_relu(model, torch.nn.functional.leaky_relu)
        layer_type = ToyReluSurrogate
    elif fgi_type == "ELU":
        toy_fgi_relu(model, torch.nn.functional.elu)
        layer_type = ToyReluSurrogate
    elif fgi_type == "CeLU":
        toy_fgi_relu(model, torch.nn.functional.celu)
        layer_type = ToyReluSurrogate
    elif fgi_type == "GeLU":
        toy_fgi_relu(model, torch.nn.functional.gelu)
        layer_type = ToyReluSurrogate
    elif fgi_type == "SeLU":
        toy_fgi_relu(model, torch.nn.functional.selu)
        layer_type = ToyReluSurrogate
    elif fgi_type == "Mish":
        toy_fgi_relu(model, torch.nn.functional.mish)
        layer_type = ToyReluSurrogate
    elif fgi_type == "Custom":  # Our B-SiLU formulation
        toy_fgi_relu(model, toy_custom_func)
        layer_type = ToyReluSurrogate
    else:
        raise ValueError(f"Invalid FGI type {fgi_type}")
    return model, layer_type


# ===========================================================================
# Shared test helpers
# ===========================================================================

ZOO_TYPES = [
    ("leakyrelu", 0.1), ("leaky", 0.1), ("elu", 0.1), ("celu", 0.1),
    ("gelu", 0.1), ("selu", 0.1), ("mish", 0.1), ("swish", 0.1),
    ("bsilu", 0.1),
    ("nelu", 0.01), ("nelu", 0.05), ("nelu", 0.1),
    ("noise_relu", 0.01), ("noise_relu", 0.1),
]

TOY_TYPES = ["NONE", "LeakyReLU", "ELU", "CeLU", "GeLU", "SeLU", "Mish", "Custom"]

# reference class name -> package class name (for cross-implementation tree checks)
REF_TO_PKG_CLASS = {
    "CnnReluSurrogate": "ReluSurrogate",
    "ToyReluSurrogate": "ReluSurrogate",
    "MobilenetSUGAR_NeLU": "SUGAR_NeLU",
    "MobilenetSUGAR_BSiLU": "SUGAR_BSiLU",
    "MobilenetSUGAR_LeakyReLU": "SUGAR_LeakyReLU",
}


def out_and_grad(module, x):
    x = x.detach().clone().requires_grad_(True)
    y = module(x)
    g = torch.autograd.grad(y.sum(), x)[0]
    return y.detach(), g


def tree(model):
    return [(name, type(mod).__name__) for name, mod in model.named_modules()]


def normalized_tree(model, mapping):
    """Module tree with class names mapped (e.g. reference names -> package names)."""
    return [(name, mapping.get(cls, cls)) for name, cls in tree(model)]


