"""Walker semantics and the high-level entry points (apply_sugar / wrap_activation)."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sugar
import _reference as ref
from conftest import build_relu_net, build_relu6_net, requires_dynamo


class ReLUContainer(nn.ReLU):
    """A ReLU subclass that itself contains a ReLU (walker edge case)."""

    def __init__(self):
        super().__init__()
        self.inner = nn.ReLU()


def _mixed_tree():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Sequential(nn.Linear(8, 8), nn.ReLU6()),
        nn.ReLU6(),
        nn.Linear(8, 3),
    )


def test_walker_replaces_counts_and_preserves_names():
    model = _mixed_tree()
    names_before = [n for n, _ in model.named_modules()]
    count = sugar.replace_modules(model, nn.ReLU, lambda: nn.Identity())
    assert count == 1
    assert [n for n, _ in model.named_modules()] == names_before
    assert isinstance(model[1], nn.Identity)


def test_walker_both_families():
    model = _mixed_tree()
    assert sugar.replace_modules(model, nn.ReLU6, lambda: nn.Identity()) == 2
    assert sugar.replace_modules(model, nn.ReLU, lambda: nn.Identity()) == 1


def test_walker_bottom_up_on_matching_container():
    model = nn.Sequential(ReLUContainer())
    count = sugar.replace_modules(model, nn.ReLU, lambda: nn.Identity())
    # descendants of a matching container are replaced first, then the container
    assert count == 2
    assert isinstance(model[0], nn.Identity)


def test_apply_sugar_auto_target(small_relu_net, small_relu6_net):
    assert sugar.apply_sugar(small_relu_net, "gelu") == 2
    assert all(isinstance(m, sugar.ReluSurrogate) for m in small_relu_net.modules()
               if isinstance(m, sugar.SugarLayer))

    assert sugar.apply_sugar(small_relu6_net, "nelu", alpha=0.05) == 2
    assert all(isinstance(m, sugar.SUGAR_NeLU) for m in small_relu6_net.modules()
               if isinstance(m, sugar.SugarLayer))


def test_apply_sugar_auto_prefers_relu6_when_both_present():
    model = nn.Sequential(nn.ReLU(), nn.ReLU6())
    assert sugar.apply_sugar(model, "nelu", alpha=0.05) == 1
    assert isinstance(model[0], nn.ReLU)           # untouched
    assert isinstance(model[1], sugar.SUGAR_NeLU)  # replaced


def test_apply_sugar_no_target_modules_returns_zero():
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
    assert sugar.apply_sugar(model, "gelu") == 0


def test_apply_sugar_explicit_target():
    model = nn.Sequential(nn.ReLU(), nn.ReLU6())
    assert sugar.apply_sugar(model, "gelu", target=nn.ReLU) == 1
    assert isinstance(model[0], sugar.ReluSurrogate)
    assert isinstance(model[1], nn.ReLU6)


def test_apply_sugar_relu_target_forward_unchanged():
    base, treated = build_relu_net(), build_relu_net()
    sugar.apply_sugar(treated, "gelu")
    x = torch.randn(7, 4)
    with torch.no_grad():
        assert torch.equal(base(x), treated(x))


def test_apply_sugar_relu6_forward_unchanged():
    base, treated = build_relu6_net(), build_relu6_net()
    sugar.apply_sugar(treated, "bsilu")
    x = torch.randn(7, 4) * 3.0  # include values above 6
    with torch.no_grad():
        assert torch.equal(base(x), treated(x))


def test_apply_sugar_bsilu_matches_function():
    model = build_relu_net()
    assert sugar.apply_sugar(model, "bsilu") == 2
    x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    actual = model[1](x)
    assert torch.equal(actual, F.relu(x))
    grad, = torch.autograd.grad(actual.sum(), x)
    expected, = torch.autograd.grad(sugar.bsilu(x).sum(), x)
    assert torch.equal(grad, expected)


@pytest.mark.parametrize("fgi_type, alpha", [
    ("leaky", 0.1), ("gelu", 0.1), ("bsilu", 0.1),
    ("nelu", 0.01), ("noise_relu", 0.01),
])
def test_apply_sugar_matches_facade(fgi_type, alpha):
    a, b = build_relu_net(), build_relu_net()
    sugar.perform_relu_fgi(a, fgi_type, alpha=alpha, compile_layer=False)
    sugar.apply_sugar(b, fgi_type, alpha=alpha, target=nn.ReLU)
    assert ref.tree(a) == ref.tree(b)
    x = torch.randn(6, 4)
    if fgi_type == "noise_relu":
        torch.manual_seed(11)
        ya = a(x)
        torch.manual_seed(11)
        yb = b(x)
    else:
        ya, yb = a(x), b(x)
    assert torch.equal(ya, yb)


def test_apply_sugar_custom_variants():
    def my_g(x):
        return torch.tanh(x) + 1.0

    a = build_relu_net()
    sugar.apply_sugar(a, sugar.Surrogate(F.relu, my_g))
    assert isinstance(a[1], sugar.Surrogate) and a[1] is not a[3]

    b = build_relu_net()
    sugar.apply_sugar(b, lambda: sugar.ReluSurrogate(my_g))
    assert isinstance(b[1], sugar.ReluSurrogate)

    x = torch.randn(6, 4)
    assert torch.equal(a(x), b(x))


def test_apply_sugar_compile_structural(small_relu_net):
    from torch._dynamo.eval_frame import OptimizedModule

    count = sugar.apply_sugar(small_relu_net, "gelu", compile_layer=True)
    assert count == 2
    layers = [m for m in small_relu_net.modules() if isinstance(m, OptimizedModule)]
    assert len(layers) == 2
    assert all(isinstance(m._orig_mod, sugar.ReluSurrogate) for m in layers)
    assert all(sugar.sugar_layer_filter(m) for m in layers)


@requires_dynamo
def test_apply_sugar_compile_eager_execution(small_relu_net, compile_eager):
    sugar.apply_sugar(small_relu_net, "gelu", compile_layer=True)
    x = torch.randn(6, 4)
    y = small_relu_net(x)
    base = build_relu_net()
    with torch.no_grad():
        assert torch.equal(base(x), y.detach())


def test_wrap_activation_baselines():
    model = build_relu_net()
    count = sugar.wrap_activation(model, F.gelu, target=nn.ReLU)
    assert count == 2
    assert all(isinstance(m, sugar.ActivationWrapper) for m in model.modules()
               if isinstance(m, sugar.SugarLayer))

    old_style = build_relu_net()
    Surrogate = ref.make_cnn_train_surrogate(F.gelu)
    ref.cnn_train_replace_layers(old_style, nn.ReLU, lambda: Surrogate())
    assert ref.normalized_tree(old_style, {"Surrogate": "ActivationWrapper"}) == ref.tree(model)
    x = torch.randn(6, 4)
    assert torch.equal(model(x), old_style(x))


def test_walker_equivalent_to_old_train_walker_on_call_sites(vgg16):
    """For the experiment call sites (leaf ReLUs / OptimizedModule wrappers) the
    package walker and cnn/train.py's replace-or-descend walker agree."""
    from torch._dynamo.eval_frame import OptimizedModule

    # swap-back scenario, compiled layers
    prepared = vgg16
    sugar.perform_relu_fgi(prepared, "bsilu", compile_layer=True)
    a, b = copy.deepcopy(prepared), copy.deepcopy(prepared)
    Surrogate = ref.make_cnn_train_surrogate(F.gelu)
    ref.cnn_train_replace_layers(a, OptimizedModule, lambda: Surrogate())
    sugar.replace_modules(b, OptimizedModule, lambda: sugar.ActivationWrapper(F.gelu))
    assert ref.normalized_tree(a, {"Surrogate": "ActivationWrapper"}) == ref.tree(b)

    # swap-back scenario, uncompiled layers
    c = build_vgg16()
    c, layer_type = sugar.perform_relu_fgi(c, "bsilu", compile_layer=False)
    d = build_vgg16()
    d, _ = sugar.perform_relu_fgi(d, "bsilu", compile_layer=False)
    ref.cnn_train_replace_layers(c, layer_type, lambda: Surrogate())
    sugar.replace_modules(d, layer_type, lambda: sugar.ActivationWrapper(F.gelu))
    assert ref.normalized_tree(c, {"Surrogate": "ActivationWrapper"}) == ref.tree(d)

    # plain ReLU leaves (baseline path)
    e, f = build_vgg16(), build_vgg16()
    ref.cnn_train_replace_layers(e, nn.ReLU, lambda: Surrogate())
    sugar.wrap_activation(f, F.gelu, target=nn.ReLU)
    assert ref.normalized_tree(e, {"Surrogate": "ActivationWrapper"}) == ref.tree(f)


def build_vgg16():
    from _models import VGG16
    torch.manual_seed(0)
    return VGG16(num_classes=100).eval()


def test_models_state_and_names_preserved(vgg16, resnet18_cifar, mobilenet_v2):
    for model, variant, target, alpha in (
        (vgg16, "bsilu", nn.ReLU, None),
        (resnet18_cifar, "nelu", nn.ReLU, 0.01),
        (mobilenet_v2, "bsilu", nn.ReLU6, None),
    ):
        keys_before = list(model.state_dict().keys())
        names_before = [n for n, _ in model.named_modules()]
        count = sugar.apply_sugar(model, variant, alpha=alpha, target=target)
        assert count > 0
        assert list(model.state_dict().keys()) == keys_before
        assert [n for n, _ in model.named_modules()] == names_before


def test_apply_sugar_errors(small_relu_net):
    with pytest.raises(ValueError, match="No SUGAR variant"):
        sugar.apply_sugar(small_relu_net, "nonexistent")
    with pytest.raises(ValueError, match="target must be"):
        sugar.apply_sugar(small_relu_net, "gelu", target="bogus")
    with pytest.raises(TypeError):
        sugar.apply_sugar(small_relu_net, 123)


def test_register_variant(small_relu_net, monkeypatch):
    builder = lambda alpha=None: sugar.ReluSurrogate(F.softplus)
    monkeypatch.setitem(sugar.surrogates._REGISTRY, ("softplus_test", nn.ReLU), builder)
    assert sugar.apply_sugar(small_relu_net, "softplus_test") == 2
    x = torch.randn(6, 4)
    with torch.no_grad():
        assert torch.equal(small_relu_net(x), build_relu_net()(x))


def test_walker_accepts_callable_condition():
    model = _mixed_tree()
    count = sugar.replace_modules(
        model, lambda mod: isinstance(mod, nn.ReLU6), lambda: nn.Identity()
    )
    assert count == 2
    assert isinstance(model[3], nn.Identity)


def test_walker_rejects_bad_condition():
    with pytest.raises(TypeError):
        sugar.replace_modules(_mixed_tree(), 123, lambda: nn.Identity())


def _cnn_layer_filter(module):
    return sugar.sugar_layer_filter(module) or isinstance(module, nn.ReLU)


@pytest.mark.parametrize("fgi_type, use_fgi, compiled", [
    ("bsilu", True, False),
    ("bsilu", True, True),
    ("relu", True, False),       # FGI passthrough (shared ReLU baseline)
    ("gelu", False, False),      # non-FGI baseline run
])
def test_cnn_flow_matches_old_flow(fgi_type, use_fgi, compiled):
    """The migrated cnn integration (apply_sugar + one layer condition for
    hooks and the eval-time swap-back) touches exactly the same modules as the
    original (perform_relu_fgi + isinstance(layer_type)) pipeline."""
    # ---- original flow
    torch.manual_seed(0)
    old_model = build_vgg16()
    if use_fgi:
        old_model, layer_type = ref.cnn_perform_relu_fgi(old_model, fgi_type,
                                                         compile_layer=compiled)
    else:
        Surrogate = ref.make_cnn_train_surrogate(F.gelu)
        ref.cnn_train_replace_layers(old_model, nn.ReLU, lambda: Surrogate())
        layer_type = Surrogate
    old_hook_names = [n for n, m in old_model.named_modules()
                      if isinstance(m, layer_type)]

    # ---- migrated flow
    torch.manual_seed(0)
    new_model = build_vgg16()
    if use_fgi:
        if fgi_type not in ("NONE", "relu", "none"):
            sugar.apply_sugar(new_model, fgi_type, target=nn.ReLU,
                               compile_layer=compiled)
    else:
        sugar.wrap_activation(new_model, F.gelu, target=nn.ReLU)
    new_hook_names = [n for n, m in new_model.named_modules()
                      if _cnn_layer_filter(m)]

    assert old_hook_names == new_hook_names

    # ---- eval-time swap-back equivalence
    Surrogate2 = ref.make_cnn_train_surrogate(F.leaky_relu)
    ref.cnn_train_replace_layers(old_model, layer_type, lambda: Surrogate2())
    sugar.replace_modules(new_model, _cnn_layer_filter,
                           lambda: sugar.ActivationWrapper(F.leaky_relu))
    assert (ref.normalized_tree(old_model, {"Surrogate": "ActivationWrapper"})
            == ref.tree(new_model))
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(old_model(x), new_model(x))



@pytest.mark.parametrize("target", [nn.ReLU, nn.ReLU6])
def test_shared_activation_slots_are_all_replaced(target):
    shared = target()
    model = nn.Sequential(shared, shared)
    count = sugar.apply_sugar(model, "nelu", target=target)
    assert count == 1
    assert model[0] is model[1]
    assert isinstance(model[0], sugar.SugarLayer)
    assert not any(isinstance(module, target) for module in model.modules())


def test_shared_nested_container_and_activation():
    shared = nn.ReLU()
    container = nn.Sequential(shared)
    model = nn.ModuleDict({"a": container, "b": container, "c": shared})
    assert sugar.apply_sugar(model, "bsilu") == 1
    assert model["a"] is model["b"]
    assert model["a"][0] is model["c"]
    assert isinstance(model["c"], sugar.SugarLayer)


def test_shared_activation_baseline_replacement():
    shared = nn.ReLU()
    model = nn.Sequential(shared, shared)
    assert sugar.wrap_activation(model, F.gelu) == 1
    assert model[0] is model[1]
    assert isinstance(model[0], sugar.ActivationWrapper)
