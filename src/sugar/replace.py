"""Module replacement: the recursive walker plus the high-level entry points.

Design rule of thumb for this repo: *recursive, parent-aware walks* for
structure surgery (you need the parent to ``setattr``), *flat iterative walks*
over ``named_modules()`` for observation (hooks, counting, listing).
"""

import copy
from typing import Callable, Optional, Type, Union

import torch
import torch.nn as nn

from sugar.surrogates import _REGISTRY, ActivationWrapper

__all__ = ["replace_modules", "apply_sugar", "wrap_activation"]

AUTO = "auto"


def replace_modules(m: nn.Module,
                     old_cls: Union[Type[nn.Module], Callable[[nn.Module], bool]],
                     new_ctor: Callable[[], nn.Module]) -> int:
    """
    Recursively replace all modules matching `old_cls` in `m`, in place,
    with `new_ctor()`. Returns the number of distinct modules replaced.
    Shared module references are preserved and all their slots are updated.

    `old_cls` may be an ``nn.Module`` subclass (matched with ``isinstance``)
    or a callable ``(module) -> bool`` condition, e.g.
    ``sugar.sugar_layer_filter``.

    Descends depth-first into every child (snapshotting the children first so
    the walk is safe against mutation), then replaces matching children.  A
    matching module that itself contains matches is processed bottom-up: its
    descendants are replaced first, then the container itself.  Module names
    (as reported by ``named_modules()``) are preserved.
    """
    if isinstance(old_cls, type):
        matches = lambda module: isinstance(module, old_cls)
    elif callable(old_cls):
        matches = old_cls
    else:
        raise TypeError(
            "old_cls must be an nn.Module subclass or a callable (module) -> bool condition"
        )
    count = 0
    memo = {id(m): m}

    def visit(parent):
        nonlocal count
        # named_children() deduplicates shared modules. Visit every registered
        # slot instead, reusing one replacement for all aliases of a module.
        for name, child in list(parent._modules.items()):
            if child is None:
                continue
            key = id(child)
            if key not in memo:
                memo[key] = child
                visit(child)
                if matches(child):
                    memo[key] = new_ctor()
                    count += 1
            setattr(parent, name, memo[key])

    visit(m)
    return count


def _resolve_target(model: nn.Module,
                    target: Union[str, Type[nn.Module], None]) -> Optional[Type[nn.Module]]:
    """Map the ``target`` argument to the activation class to replace."""
    if target is None or (isinstance(target, str) and target == AUTO):
        # Auto-detect from what the model actually contains; prefer the
        # ReLU6 family (MobileNet-style) when both are present.
        for cls in (nn.ReLU6, nn.ReLU):
            if any(isinstance(mod, cls) for mod in model.modules()):
                return cls
        return None
    if isinstance(target, str):
        raise ValueError(
            f"target must be '{AUTO}', None, or an nn.Module subclass; got {target!r}"
        )
    if not (isinstance(target, type) and issubclass(target, nn.Module)):
        raise TypeError(
            f"target must be '{AUTO}', None, or an nn.Module subclass; got {target!r}"
        )
    return target


def _resolve_builder(variant, old_cls, alpha) -> Callable[[], nn.Module]:
    """Turn the ``variant`` argument into a zero-argument module factory."""
    if isinstance(variant, str):
        try:
            builder = _REGISTRY[(variant, old_cls)]
        except KeyError:
            supported = sorted({name for (name, _) in _REGISTRY})
            raise ValueError(
                f"No SUGAR variant {variant!r} for target {old_cls.__name__}. "
                f"Registered variants: {supported}"
            ) from None
        return lambda: builder(alpha)
    if isinstance(variant, type) and issubclass(variant, nn.Module):
        return variant
    if isinstance(variant, nn.Module):
        template = variant
        return lambda: copy.deepcopy(template)
    if callable(variant):
        return variant
    raise TypeError(
        "variant must be a registered name, an nn.Module subclass, a zero-argument "
        f"factory, or an nn.Module template; got {type(variant)!r}"
    )


def apply_sugar(model: nn.Module,
                 variant,
                 *,
                 alpha=None,
                 target: Union[str, Type[nn.Module], None] = AUTO,
                 compile_layer: bool = False) -> int:
    """Replace ReLU-like activations in *model* with SUGAR surrogates, in place.

    This is the one-call integration point: call it once on a freshly built
    model, before training.  The training loop itself needs no changes.

    Parameters
    ----------
    model:
        The model to modify in place.
    variant:
        Either the name of a registered variant (``"nelu"``, ``"bsilu"``,
        ``"leaky"``, ``"gelu"``, ...; both target families support the same
        built-in variants), an ``nn.Module`` subclass (instantiated
        with no arguments per slot), a zero-argument factory callable, or an
        ``nn.Module`` instance used as a template (deep-copied per slot, e.g.
        ``Surrogate(torch.nn.functional.relu, my_g)``). Custom modules define
        their own forward function and saturation; target only selects slots.
    alpha:
        Parameter for variants that use one (``nelu``/
        ``noise_relu``/``leaky``).  ``None`` uses the variant's own default.
    target:
        Which activation class to replace: ``"auto"`` picks ``nn.ReLU6`` if the
        model contains any, else ``nn.ReLU``.  Pass the class explicitly
        (e.g. ``torch.nn.ReLU``) to override.
    compile_layer:
        Wrap each replacement module in ``torch.compile``.

    Returns
    -------
    int
        The number of distinct modules replaced (0 if no child modules match
        the target class). Already replaced SUGAR layers are not retargeted.
    """
    old_cls = _resolve_target(model, target)
    if old_cls is None:
        return 0
    factory = _resolve_builder(variant, old_cls, alpha)

    def _factory():
        layer = factory()
        if not isinstance(layer, nn.Module):
            raise TypeError(
                f"variant factory must produce an nn.Module, got {type(layer)!r}"
            )
        if compile_layer:
            # Mark the inner module so hook filters (sugar_layer_filter) match
            # the wrapper only.
            layer._sugar_compiled_inner = True
            layer = torch.compile(layer)
        return layer

    return replace_modules(model, old_cls, _factory)


def wrap_activation(model: nn.Module,
                     fn: Callable,
                     target: Union[str, Type[nn.Module], None] = AUTO) -> int:
    """Replace plain ReLU-like activations, in place, with a parameterless ``fn`` wrapper.

    Used for non-FGI baseline runs: the model keeps the same module structure
    (same names, same hook attachment sets) but its activations compute ``fn``
    directly.  Returns the number of replacements made.
    """
    old_cls = _resolve_target(model, target)
    if old_cls is None:
        return 0
    return replace_modules(model, old_cls, lambda: ActivationWrapper(fn))

