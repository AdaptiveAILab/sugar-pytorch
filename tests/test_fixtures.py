"""Permanent parity net: the sugar package vs. golden fixtures.

tests/fixtures/golden.pt records the behaviour of the original experiment
code (via the reference copies in tests/_reference.py, applied to
the model architectures vendored in tests/_models.py); these tests pin the
package to it.  The mobilenet nelu entry is recorded from the fixed
SUGAR_NeLU (F.relu6 forward and zero gradient at x >= 6; the original uncapped F.relu forward was a bug).
The toy adapter itself (experiments/toy/relu_fgi.py) is pinned in the
experiments repo; here the package is pinned to the original toy math.

The fixtures are torch-build-specific and therefore not committed; the
``golden`` fixture regenerates them (via tests/generate_fixtures.py)
whenever they are missing or were recorded under a different torch build.
Set ``SUGAR_NO_FIXTURE_REGEN=1`` to disable automatic regeneration.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sugar
from _models import VGG16, resnet18_cifar

TESTS_DIR = Path(__file__).resolve().parent

FIXTURES = TESTS_DIR / "fixtures" / "golden.pt"

# trees recorded by the old code use the old local wrapper class name
PKG_TO_OLD = {"ActivationWrapper": "Surrogate"}


def _tree(model):
    return [(name, PKG_TO_OLD.get(type(mod).__name__, type(mod).__name__))
            for name, mod in model.named_modules()]


def _record(model, x, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    h = x.detach().clone().requires_grad_(True)
    y = model(h)
    y.sum().backward()
    return y.detach().clone(), h.grad.detach().clone()


def build_vgg16():
    torch.manual_seed(0)
    return VGG16(num_classes=100).eval()


def build_resnet18():
    try:
        torch.manual_seed(0)
        return resnet18_cifar(num_classes=100).eval()
    except ImportError:
        pytest.skip("torchvision not available")


def build_mobilenet_v2():
    from torchvision.models import mobilenet_v2
    torch.manual_seed(0)
    return mobilenet_v2(weights=None).eval()


def build_toy_net():
    torch.manual_seed(4)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16),
                         nn.ReLU(), nn.Linear(16, 4))


def _load_golden():
    return torch.load(FIXTURES, map_location="cpu", weights_only=False)


def _stale(data):
    return (data is None or data["meta"].get("schema") != 3
            or data["meta"]["torch_version"] != torch.__version__)


@pytest.fixture(scope="module")
def golden():
    """Golden data for the current torch build, regenerating the fixtures
    first if they are missing or were recorded under a different build."""
    data = _load_golden() if FIXTURES.exists() else None
    regen_error = None
    if _stale(data) and not os.environ.get("SUGAR_NO_FIXTURE_REGEN"):
        try:
            import generate_fixtures
            generate_fixtures.main()
            data = _load_golden()
        except Exception as exc:  # e.g. read-only checkout or a missing optional dep
            regen_error = exc
    if _stale(data):
        recorded = None if data is None else data["meta"]["torch_version"]
        reason = (f"golden fixtures unavailable for torch {torch.__version__} "
                  f"(recorded: {recorded}); run python tests/generate_fixtures.py")
        if regen_error is not None:
            reason += f" (automatic regeneration failed: {regen_error!r})"
        pytest.skip(reason)
    return data


def test_vgg16_fgi_parity(golden):
    g = golden["vgg16"]
    x = g["input"]
    for fgi_type, alpha in (("bsilu", 0.1), ("nelu", 0.01), ("noise_relu", 0.01)):
        exp = g["fgi"][fgi_type]
        model = build_vgg16()
        _, layer_type = sugar.perform_relu_fgi(model, fgi_type, alpha=alpha,
                                               compile_layer=False)
        assert layer_type.__name__ == exp["layer_type"]
        assert _tree(model) == exp["tree"]
        out, grad = _record(model, x, seed=77 if fgi_type == "noise_relu" else None)
        assert torch.equal(out, exp["out"])
        assert torch.equal(grad, exp["grad"])

    # compiled: structure only (inductor execution needs a C++ toolchain)
    exp = g["fgi"]["bsilu_compiled"]
    model = build_vgg16()
    _, layer_type = sugar.perform_relu_fgi(model, "bsilu", compile_layer=True)
    assert layer_type.__name__ == exp["layer_type"]
    assert _tree(model) == exp["tree"]


def test_vgg16_baseline_and_swapback_parity(golden):
    g = golden["vgg16"]
    x = g["input"]

    exp = g["baseline"]["gelu"]
    model = build_vgg16()
    n = sugar.wrap_activation(model, F.gelu, target=nn.ReLU)
    assert n == g["relu_count"]
    assert _tree(model) == exp["tree"]
    out, grad = _record(model, x)
    assert torch.equal(out, exp["out"])
    assert torch.equal(grad, exp["grad"])

    exp = g["swapback"]["leaky"]
    model = build_vgg16()
    model, layer_type = sugar.perform_relu_fgi(model, "bsilu", compile_layer=False)
    sugar.replace_modules(model, layer_type, lambda: sugar.ActivationWrapper(F.leaky_relu))
    assert _tree(model) == exp["tree"]
    out, _ = _record(model, x)
    assert torch.equal(out, exp["out"])


def test_resnet18_fgi_parity(golden):
    g = golden["resnet18"]
    exp = g["fgi"]["bsilu"]
    model = build_resnet18()
    sugar.perform_relu_fgi(model, "bsilu", compile_layer=False)
    assert _tree(model) == exp["tree"]
    out, grad = _record(model, g["input"])
    assert torch.equal(out, exp["out"])
    assert torch.equal(grad, exp["grad"])


def test_mobilenet_v2_parity(golden):
    g = golden["mobilenet_v2"]
    x = g["input"]
    for variant, alpha in (("nelu", 0.05), ("bsilu", None), ("leaky", 0.01)):
        exp = g["variants"][variant]
        model = build_mobilenet_v2()
        n = sugar.apply_sugar(model, variant, alpha=alpha, target=nn.ReLU6)
        assert n == exp["count"]
        assert _tree(model) == exp["tree"]
        assert sorted(sugar.list_activations(model)) == exp["list_activations"]
        out, grad = _record(model, x)
        assert torch.equal(out, exp["out"])
        assert torch.equal(grad, exp["grad"])


# original toy type -> package equivalent; the adapter experiments/toy/relu_fgi.py
# maps these the same way (the adapter itself is pinned in the experiments repo)
TOY_TO_SUGAR = {
    "NONE": None,
    "LeakyReLU": "leaky",
    "ELU": "elu",
    "CeLU": "celu",
    "GeLU": "gelu",
    "SeLU": "selu",
    "Mish": "mish",
}


def _toy_custom(x):
    # the toy's own B-SiLU formulation, mathematically identical to bsilu
    # (kept in the toy's op order so the comparison stays bit-exact)
    return (x + 1.67) * torch.sigmoid(x) - 0.835


def test_toy_parity(golden):
    g = golden["toy"]
    for fgi, exp in g["types"].items():
        model = build_toy_net()
        if fgi == "Custom":
            sugar.apply_sugar(model, lambda: sugar.ReluSurrogate(_toy_custom),
                              target=nn.ReLU)
        elif TOY_TO_SUGAR[fgi] is not None:
            sugar.apply_sugar(model, TOY_TO_SUGAR[fgi], target=nn.ReLU)
        assert _tree(model) == exp["tree"]
        out, grad = _record(model, g["input"])
        assert torch.equal(out, exp["out"])
        assert torch.equal(grad, exp["grad"])


def test_noise_stream_parity(golden):
    g = golden["noise_stream"]
    model = build_vgg16()
    sugar.perform_relu_fgi(model, "noise_relu", alpha=0.01, compile_layer=False)
    torch.manual_seed(99)
    outs = [model(golden["vgg16"]["input"]).detach().clone() for _ in range(3)]
    assert all(torch.equal(a, b) for a, b in zip(outs, g["outputs"]))


def test_state_dict_keys_parity(golden):
    assert list(build_vgg16().state_dict().keys()) == golden["vgg16"]["state_dict_keys"]
    assert list(build_resnet18().state_dict().keys()) == golden["resnet18"]["state_dict_keys"]
    assert list(build_mobilenet_v2().state_dict().keys()) == golden["mobilenet_v2"]["state_dict_keys"]

