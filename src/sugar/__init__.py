"""SUGAR -- surrogate-gradient (FGI) replacements for ReLU-like activations.

Replace the backward pass of an existing model's ReLU-family activations with
a surrogate gradient while keeping the forward pass unchanged.
Call ``sugar.apply_sugar(model, "bsilu")`` once on a freshly built model,
then train as usual. See the README for independent usage examples.
Most variants use the derivative of a surrogate; noise injection directly
specifies random backward slopes for non-positive inputs.
"""

from sugar.surrogates import (
    ActivationWrapper,
    ReluSurrogate,
    SugarLayer,
    Surrogate,
    SUGAR_BSiLU,
    SUGAR_LeakyReLU,
    SUGAR_NeLU,
    nelu,
    noise_relu,
    bsilu,
    register_variant,
)
from sugar.replace import apply_sugar, replace_modules, wrap_activation
from sugar.fgi import get_surrogate_layer, perform_relu_fgi
from sugar.inspection import disable_inplace, list_activations, sugar_layer_filter

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # surrogate modules
    "SugarLayer",
    "Surrogate",
    "ReluSurrogate",
    "ActivationWrapper",
    "SUGAR_NeLU",
    "SUGAR_BSiLU",
    "SUGAR_LeakyReLU",
    # surrogate functions
    "nelu",
    "noise_relu",
    "bsilu",
    "register_variant",
    # replacement
    "replace_modules",
    "apply_sugar",
    "wrap_activation",
    # compatibility façade (the paper's API)
    "perform_relu_fgi",
    "get_surrogate_layer",
    # inspection
    "list_activations",
    "sugar_layer_filter",
    "disable_inplace",
]
