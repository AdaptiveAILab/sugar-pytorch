"""Compatibility façade for the paper's ``perform_relu_fgi`` API.

New code should prefer ``sugar.apply_sugar``.
"""

import torch
from torch._dynamo.eval_frame import OptimizedModule
from torch.nn import ReLU

from sugar.replace import replace_modules
from sugar.surrogates import ReluSurrogate, nelu, noise_relu, bsilu

__all__ = ["perform_relu_fgi", "get_surrogate_layer"]


def get_surrogate_layer(g, *args, compile_layer=False):
    """Return a factory creating ``ReluSurrogate(g, *args)`` layers.

    ``factory.layer_type()`` reports the class of the produced layers
    (``torch._dynamo.eval_frame.OptimizedModule`` when ``compile_layer=True``)
    so callers can find the replaced modules again with ``isinstance`` checks.
    """
    holder = {}

    def _factory():
        layer = ReluSurrogate(g, *args)
        if compile_layer:
            layer._sugar_compiled_inner = True  # hook filters target the wrapper
            layer = torch.compile(layer)
            holder["cls"] = OptimizedModule
        else:
            holder["cls"] = ReluSurrogate
        return layer

    _factory.layer_type = lambda: holder["cls"]
    return _factory


def perform_relu_fgi(model, fgi_type="relu", alpha=0.1, compile_layer=False):
    """Replace every ``nn.ReLU`` in *model* with a surrogate-gradient module.

    Returns ``(model, layer_type)`` where ``layer_type`` is the class of the
    replacement modules (``torch._dynamo.eval_frame.OptimizedModule`` when
    ``compile_layer=True``).  For ``fgi_type`` in ``("NONE", "relu", "none")``
    the model is returned unchanged with ``layer_type = torch.nn.ReLU``.

    Note: ``compile_layer`` defaults to ``False`` here; the CNN experiment
    passes ``compile_layer=True``.
    """
    if fgi_type in ("NONE", "relu", "none"):
        return (model, torch.nn.ReLU)
    if fgi_type in ("leakyrelu", "leaky"):
        g = lambda x: torch.nn.functional.leaky_relu(x, negative_slope=alpha)
    elif fgi_type == "elu":
        g = torch.nn.functional.elu
    elif fgi_type == "celu":
        g = torch.nn.functional.celu
    elif fgi_type == "gelu":
        g = torch.nn.functional.gelu
    elif fgi_type == "selu":
        g = torch.nn.functional.selu
    elif fgi_type == "mish":
        g = torch.nn.functional.mish
    elif fgi_type == "swish":
        g = torch.nn.functional.silu
    elif fgi_type == "nelu":
        g = lambda x: nelu(x, alpha=alpha)
    elif fgi_type == "bsilu":
        g = lambda x: bsilu(x)
    elif fgi_type == "noise_relu":
        g = lambda x: noise_relu(x, std=alpha)
    else:
        raise ValueError(f"Invalid FGI type {fgi_type}")
    factory = get_surrogate_layer(g, compile_layer=compile_layer)
    replace_modules(model, ReLU, factory)
    layer_type = factory.layer_type()
    return (model, layer_type)
