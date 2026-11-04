"""Legacy façade parity with the original cnn perform_relu_fgi."""

import pytest
import torch
import torch.nn as nn

import sugar
import _reference as ref
from conftest import build_relu_net, build_vgg16, requires_dynamo


@pytest.mark.parametrize("passthrough", ["NONE", "relu", "none"])
def test_passthrough(passthrough, small_relu_net):
    before = ref.tree(small_relu_net)
    model, layer_type = sugar.perform_relu_fgi(small_relu_net, passthrough)
    assert model is small_relu_net
    assert layer_type is nn.ReLU
    assert ref.tree(model) == before


def _grads(model, x, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    h = x.detach().clone().requires_grad_(True)
    model(h).sum().backward()
    return h.grad


@pytest.mark.parametrize("fgi_type, alpha", ref.ZOO_TYPES)
def test_facade_matches_old_implementation(fgi_type, alpha):
    old_model, new_model = build_vgg16(), build_vgg16()
    old_model, old_lt = ref.cnn_perform_relu_fgi(old_model, fgi_type, alpha=alpha,
                                                 compile_layer=False)
    new_model, new_lt = sugar.perform_relu_fgi(new_model, fgi_type, alpha=alpha,
                                               compile_layer=False)
    assert ref.REF_TO_PKG_CLASS[old_lt.__name__] == new_lt.__name__ == "ReluSurrogate"
    assert ref.normalized_tree(old_model, ref.REF_TO_PKG_CLASS) == ref.tree(new_model)
    assert list(old_model.state_dict().keys()) == list(new_model.state_dict().keys())

    x = torch.randn(2, 3, 32, 32)
    seed = 21 if fgi_type == "noise_relu" else None
    if seed is not None:
        torch.manual_seed(seed)
        y1 = old_model(x)
        torch.manual_seed(seed)
        y2 = new_model(x)
    else:
        y1, y2 = old_model(x), new_model(x)
    assert torch.equal(y1, y2)
    assert torch.equal(_grads(old_model, x, 22 if fgi_type == "noise_relu" else None),
                       _grads(new_model, x, 22 if fgi_type == "noise_relu" else None))


def test_facade_invalid_type(small_relu_net):
    with pytest.raises(ValueError, match="Invalid FGI type bogus"):
        sugar.perform_relu_fgi(small_relu_net, "bogus")


def test_facade_no_relu_matches_old_edge_behaviour():
    def build():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(2, 2), nn.Tanh())

    with pytest.raises(KeyError):
        ref.cnn_perform_relu_fgi(build(), "gelu", compile_layer=False)
    with pytest.raises(KeyError):
        sugar.perform_relu_fgi(build(), "gelu", compile_layer=False)


def test_facade_compile_structural(vgg16):
    from torch._dynamo.eval_frame import OptimizedModule

    relu_count = sum(isinstance(m, nn.ReLU) for m in vgg16.modules())
    model, layer_type = sugar.perform_relu_fgi(vgg16, "bsilu", compile_layer=True)
    assert layer_type is OptimizedModule
    assert sum(isinstance(m, layer_type) for m in model.modules()) == relu_count
    assert sum(sugar.sugar_layer_filter(m) for m in model.modules()) == relu_count
    assert all(isinstance(m._orig_mod, sugar.ReluSurrogate)
               for m in model.modules() if isinstance(m, OptimizedModule))


@requires_dynamo
def test_facade_compile_eager_math(small_relu_net, compile_eager):
    from torch._dynamo.eval_frame import OptimizedModule

    model, layer_type = sugar.perform_relu_fgi(small_relu_net, "gelu", compile_layer=True)
    assert layer_type is OptimizedModule
    x = torch.randn(6, 4)
    y = model(x)
    base = build_relu_net()
    with torch.no_grad():
        assert torch.equal(base(x), y.detach())


def test_get_surrogate_layer_layer_type_closure():
    import torch.nn.functional as F

    factory = sugar.get_surrogate_layer(F.gelu, compile_layer=False)
    assert isinstance(factory(), sugar.ReluSurrogate)
    assert factory.layer_type() is sugar.ReluSurrogate


def test_facade_noise_rng_stream_matches_old():
    x = torch.randn(4, 4)

    def run(model):
        torch.manual_seed(31)
        return [model(x).detach().clone() for _ in range(3)]

    old_model = build_relu_net()
    ref.cnn_perform_relu_fgi(old_model, "noise_relu", alpha=0.01, compile_layer=False)
    new_model = build_relu_net()
    sugar.perform_relu_fgi(new_model, "noise_relu", alpha=0.01, compile_layer=False)
    assert all(torch.equal(a, b) for a, b in zip(run(old_model), run(new_model)))

