"""Small helpers to inspect and prepare SUGAR-treated models.

These verify/prepare the FGI surgery itself; they are deliberately tiny.
"""

import torch.nn as nn

from sugar.surrogates import SugarLayer

__all__ = ["list_activations", "sugar_layer_filter", "disable_inplace"]


def list_activations(model: nn.Module) -> set:
    """Names of the activation module classes present in *model*.

    Recognises the plain ReLU-like torch modules plus every SUGAR layer
    (anything deriving from :class:`sugar.SugarLayer`).
    """
    types = set()
    for mod in model.modules():
        if isinstance(mod, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Hardsigmoid,
                            SugarLayer)):
            types.add(type(mod).__name__)
    return types


def sugar_layer_filter(module: nn.Module) -> bool:
    """Hook layer condition: True for SUGAR activation modules.

    Matches exactly one module per replaced slot: the ``torch.compile``
    wrapper (``torch._dynamo.eval_frame.OptimizedModule``) when the layer was
    compiled through the sugar APIs (the inner module is marked and skipped),
    or the plain SUGAR module itself otherwise.
    """
    inner = getattr(module, "_orig_mod", None)
    if inner is not None:
        return isinstance(inner, SugarLayer)
    if isinstance(module, SugarLayer):
        return not getattr(module, "_sugar_compiled_inner", False)
    return False


def disable_inplace(m: nn.Module) -> None:
    """Set ``inplace = False`` on every ``nn.ReLU`` in *m*.

    Out-of-place activations are required for forward hooks to observe outputs.
    """
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
