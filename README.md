<p align="center">
  <img src="assets/SUGAR_light.png#gh-light-mode-only"
       alt="SUGAR" width="100%">
  <img src="assets/SUGAR_dark.png#gh-dark-mode-only"
       alt="SUGAR" width="100%">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

SUGAR keeps ReLU's forward pass and replaces its backward pass with a surrogate
gradient. Apply it once to your model before training; the training loop stays
the same.

## Installation

Requires Python 3.9 or newer and PyTorch 2.0 or newer. From this repository's root:

```bash
pip install -e .
```

The package is imported as `sugar`.

## Quick start

```python
import torch
from torch import nn
import sugar

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
count = sugar.apply_sugar(model, "bsilu")
assert count == 1

# Train normally.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x, labels = torch.randn(8, 10), torch.randint(0, 2, (8,))
optimizer.zero_grad()
loss = nn.functional.cross_entropy(model(x), labels)
loss.backward()
optimizer.step()
```

## Other ways to use SUGAR

Each example below is an alternative, starting from a fresh model.

### Choose a surrogate and its parameter

```python
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
sugar.apply_sugar(model, "nelu", alpha=0.05)
```

Use `"leaky"` with `alpha` to set the negative slope. Omitting `alpha` uses
the variant's default.

### Target ReLU6 explicitly

```python
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU6(), nn.Linear(32, 2))
sugar.apply_sugar(model, "leaky", alpha=0.01, target=nn.ReLU6)
```

For all built-in ReLU6 variants, the forward output saturates at 6 and the
backward gradient is zero for inputs at or above 6.

### Supply a custom surrogate

```python
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
sugar.apply_sugar(model, sugar.Surrogate(nn.functional.relu, torch.tanh))
```

`Surrogate(f, g)` uses `f` for forward values and the derivative of `g` for
backpropagation. Custom surrogates define their own saturation behavior.

## Supported variants

Both `nn.ReLU` and `nn.ReLU6` support the same variants:
`bsilu`, `nelu`, `leaky`, `elu`, `celu`, `gelu`, `selu`, `mish`, `swish`,
and `noise_relu`.

Defaults: `nelu` alpha 0.05, `leaky` slope 0.01, and `noise_relu` standard
deviation 0.01 (set through `alpha`). B-SiLU uses a fixed parameter of 1.67;
`swish` is the standard SiLU surrogate.

## Behavior

- `apply_sugar` modifies the model in place and returns the number of distinct
  modules replaced (zero if none match). Shared activations remain shared.
- It replaces activation modules inside a model, not functional calls such as
  `torch.relu(x)` or `F.relu(x)`. Reapplying it does not switch existing SUGAR layers.
- Automatic targeting selects `nn.ReLU6` if present, otherwise `nn.ReLU`.
  For models containing both, specify each target in a separate call.
- Built-in replacements add no trainable parameters. Without compilation,
  they preserve module names and `state_dict()` keys, so baseline weights can
  be loaded into a model with SUGAR applied.
- Variants use indirect injection, except `noise_relu`, which directly injects
  Gaussian backward slopes for non-positive inputs and unit slopes for positive
  inputs. Noise is resampled on each forward call; use `torch.manual_seed` to
  control its random sequence.
- Pass `compile_layer=True` to compile each replacement with `torch.compile`.
- `sugar.wrap_activation(model, fn)` replaces the activation itself for a
  conventional baseline, using `fn` in both forward and backward passes.

Additional inspection, replacement and compatibility helpers are documented
in their docstrings under `src/sugar/`.

## Tests

```bash
pip install -e ".[test]"
pytest tests
```

Tests cover forward values, gradients, replacement behavior and reference-model
comparisons. Reference fixtures are generated automatically for the installed
PyTorch version.
