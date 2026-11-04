"""Golden-math parity: package surrogates vs. the original experiment code."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sugar
import _reference as ref


@pytest.mark.parametrize("fgi_type, alpha", ref.ZOO_TYPES)
def test_relu_zoo_registry_matches_old_dispatch(fgi_type, alpha, x_act):
    old = ref.CnnReluSurrogate(ref.cnn_dispatch_g(fgi_type, alpha))
    new = sugar.surrogates._REGISTRY[(fgi_type, nn.ReLU)](alpha)
    if fgi_type == "noise_relu":
        torch.manual_seed(123)
        y1, g1 = ref.out_and_grad(old, x_act)
        torch.manual_seed(123)
        y2, g2 = ref.out_and_grad(new, x_act)
    else:
        y1, g1 = ref.out_and_grad(old, x_act)
        y2, g2 = ref.out_and_grad(new, x_act)
    assert torch.equal(y1, y2)
    assert torch.equal(g1, g2)


@pytest.mark.parametrize("fgi_type, default, kwarg", [
    ("nelu", 0.05, "alpha"),
    ("noise_relu", 0.01, "std"),
])
def test_relu_zoo_alpha_none_uses_variant_default(fgi_type, default, kwarg, x_act):
    fn = {"nelu": sugar.nelu, "noise_relu": sugar.noise_relu}[fgi_type]
    expected = ref.CnnReluSurrogate(lambda x: fn(x, **{kwarg: default}))
    new = sugar.surrogates._REGISTRY[(fgi_type, nn.ReLU)](None)
    if fgi_type == "noise_relu":
        torch.manual_seed(5)
        y1, _ = ref.out_and_grad(expected, x_act)
        torch.manual_seed(5)
        y2, _ = ref.out_and_grad(new, x_act)
    else:
        y1, _ = ref.out_and_grad(expected, x_act)
        y2, _ = ref.out_and_grad(new, x_act)
    assert torch.equal(y1, y2)


def test_sugar_bsilu_parity(x_act):
    old, new = ref.MobilenetSUGAR_BSiLU(), sugar.SUGAR_BSiLU()
    y1, g1 = ref.out_and_grad(old, x_act)
    y2, g2 = ref.out_and_grad(new, x_act)
    assert torch.equal(y1, y2)
    assert torch.equal(g1, g2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alpha", [-0.01, 0.0, 0.01, 0.05, 0.1, 0.2])
def test_sugar_leaky_parity(alpha, x_act, dtype):
    x_act = torch.cat([x_act.to(dtype), torch.linspace(-10, 10, 4001, dtype=dtype)])
    old = ref.MobilenetSUGAR_LeakyReLU(alpha)
    new = sugar.SUGAR_LeakyReLU(alpha)
    y1, g1 = ref.out_and_grad(old, x_act)
    y2, g2 = ref.out_and_grad(new, x_act)
    assert torch.equal(y1, y2)
    if dtype is torch.float32:
        assert torch.equal(g1, g2)
    else:
        # The old torch.where(scalar, scalar) creates float32 slopes even
        # for float64 inputs. Indirect injection retains the input precision.
        torch.testing.assert_close(g1, g2, rtol=6e-8, atol=0)
    expected = torch.where(x_act > 0, torch.ones_like(x_act),
                           torch.full_like(x_act, alpha))
    expected[x_act >= 6] = 0
    assert torch.equal(g2, expected)


def test_sugar_relu6_family_forwards_are_relu6(x_act):
    for module in (sugar.SUGAR_NeLU(0.05), sugar.SUGAR_BSiLU(),
                   sugar.SUGAR_LeakyReLU(0.01)):
        with torch.no_grad():
            assert torch.equal(module(x_act), F.relu6(x_act))


def test_generic_surrogate_matches_relu_surrogate(x_act):
    a = sugar.Surrogate(F.relu, F.gelu)
    b = sugar.ReluSurrogate(F.gelu)
    ya, ga = ref.out_and_grad(a, x_act)
    yb, gb = ref.out_and_grad(b, x_act)
    assert torch.equal(ya, yb)
    assert torch.equal(ga, gb)
    with torch.no_grad():
        assert torch.equal(a(x_act), F.relu(x_act))


def test_generic_surrogate_relu6_target(x_act):
    module = sugar.Surrogate(F.relu6, lambda x: (x + 1.67) * torch.sigmoid(x) - 0.835)
    with torch.no_grad():
        assert torch.equal(module(x_act), F.relu6(x_act))


def test_bsilu_and_toy_custom_func_are_equivalent(x_act):
    # Mathematically identical B-SiLU formulations, but computed with a
    # different op order -> not bit-equal.  The toy adapter therefore keeps
    # the toy formulation verbatim for "Custom".
    assert torch.allclose(sugar.bsilu(x_act), ref.toy_custom_func(x_act),
                          rtol=1e-6, atol=1e-7)
    assert torch.equal(ref.cnn_bsilu(x_act), sugar.bsilu(x_act))


def test_noise_relu_rng_stream_parity(x_act):
    def stream(module):
        torch.manual_seed(7)
        return [module(x_act).detach().clone() for _ in range(3)]

    old = stream(ref.CnnReluSurrogate(ref.cnn_dispatch_g("noise_relu", 0.01)))
    new = stream(sugar.ReluSurrogate(lambda x: sugar.noise_relu(x, std=0.01)))
    assert all(torch.equal(a, b) for a, b in zip(old, new))


def test_reprs_match_old_format():
    assert repr(sugar.SUGAR_NeLU(0.05)) == "SUGAR_NeLU(alpha=0.05)"
    assert repr(sugar.SUGAR_LeakyReLU(0.01)) == "SUGAR_LeakyReLU(alpha=0.01)"
    assert str(sugar.SUGAR_BSiLU()) == "SUGAR_BSiLU()"


def test_surrogates_are_parameterless():
    modules = [
        sugar.SUGAR_NeLU(0.05), sugar.SUGAR_BSiLU(), sugar.SUGAR_LeakyReLU(0.01),
        sugar.ReluSurrogate(F.relu), sugar.Surrogate(F.relu6, F.silu),
        sugar.ActivationWrapper(F.gelu),
    ]
    for module in modules:
        assert isinstance(module, sugar.SugarLayer)
        assert len(module.state_dict()) == 0
        assert len(list(module.parameters())) == 0


def test_list_activations_reports_sugar_layers(small_relu_net, small_relu6_net):
    sugar.apply_sugar(small_relu_net, "gelu")
    sugar.apply_sugar(small_relu6_net, "bsilu")
    assert sugar.list_activations(small_relu_net) == {"ReluSurrogate"}
    assert sugar.list_activations(small_relu6_net) == {"SUGAR_BSiLU"}


def test_sugar_layer_filter_unwraps_compile(small_relu_net):
    sugar.apply_sugar(small_relu_net, "gelu", compile_layer=True)
    matched = [m for m in small_relu_net.modules() if sugar.sugar_layer_filter(m)]
    assert len(matched) == 2
    assert all(type(m).__name__ == "OptimizedModule" for m in matched)
    assert not sugar.sugar_layer_filter(nn.ReLU())



@pytest.mark.parametrize("target", [nn.ReLU, nn.ReLU6])
@pytest.mark.parametrize("alpha", [0.0, 0.01, 0.2])
def test_leaky_uses_requested_slope(target, alpha):
    model = nn.Sequential(target())
    sugar.apply_sugar(model, "leaky", alpha=alpha, target=target)
    x = torch.tensor([-2.0, 2.0, 6.0, 7.0], requires_grad=True)
    y = model(x)
    assert torch.equal(y, target()(x))
    grad, = torch.autograd.grad(y.sum(), x)
    expected = torch.tensor([alpha, 1.0, 0.0, 0.0] if target is nn.ReLU6
                            else [alpha, 1.0, 1.0, 1.0])
    torch.testing.assert_close(grad, expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["nelu", "bsilu", "leaky"])
def test_relu6_saturates_forward_and_backward(variant):
    model = nn.Sequential(nn.ReLU6())
    sugar.apply_sugar(model, variant, target=nn.ReLU6)
    x = torch.tensor([-2.0, 2.0, 5.999, 6.0, 7.0], requires_grad=True)
    y = model(x)
    assert torch.equal(y, F.relu6(x))
    grad, = torch.autograd.grad(y.sum(), x)
    assert torch.equal(grad[x >= 6], torch.zeros_like(grad[x >= 6]))
    if variant == "nelu":
        assert torch.equal(grad[(x > 0) & (x < 6)], torch.ones(2))
        expected_negative = 2 * 0.05 * x[0] / (1 + x[0].square()).square()
        torch.testing.assert_close(grad[0], expected_negative)


BUILTIN_VARIANTS = sorted(name for name, target in sugar.surrogates._REGISTRY
                          if target is nn.ReLU)


def test_builtin_target_families_have_identical_variants():
    registry = sugar.surrogates._REGISTRY
    assert {name for name, target in registry if target is nn.ReLU} == {
        name for name, target in registry if target is nn.ReLU6}


@pytest.mark.parametrize("variant", BUILTIN_VARIANTS)
def test_all_relu6_variants_match_surrogate_below_saturation(variant):
    x = torch.tensor([-3., -0.5, 0., 0.5, 5.9, 6., 7.], requires_grad=True)
    relu = nn.Sequential(nn.ReLU())
    relu6 = nn.Sequential(nn.ReLU6())
    sugar.apply_sugar(relu, variant, alpha=0.2, target=nn.ReLU)
    sugar.apply_sugar(relu6, variant, alpha=0.2, target=nn.ReLU6)
    torch.manual_seed(123)
    expected_grad, = torch.autograd.grad(relu(x).sum(), x)
    torch.manual_seed(123)
    y = relu6(x)
    actual_grad, = torch.autograd.grad(y.sum(), x)
    assert torch.equal(y, F.relu6(x))
    torch.testing.assert_close(actual_grad[x < 6], expected_grad[x < 6])
    assert torch.equal(actual_grad[x >= 6], torch.zeros_like(actual_grad[x >= 6]))


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
def test_nelu_matches_intended_formula(alpha):
    x = torch.tensor([-3., -1., 0., 1., 7.], requires_grad=True)
    gx = torch.where(x > 0, x, -alpha / (1 + x.square()))
    expected = gx - gx.detach() + F.relu(x).detach()
    model = nn.Sequential(nn.ReLU())
    sugar.apply_sugar(model, "nelu", alpha=alpha, target=nn.ReLU)
    actual = model(x)
    assert torch.equal(actual, expected)
    actual_grad, = torch.autograd.grad(actual.sum(), x)
    expected_grad, = torch.autograd.grad(expected.sum(), x)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("target", [nn.ReLU, nn.ReLU6])
@pytest.mark.parametrize("std", [0.0, 0.01, 0.2])
def test_noise_injects_seeded_slopes(target, std):
    x = torch.tensor([-3., -1., 0., 1., 5.9, 6., 7.], requires_grad=True)
    model = nn.Sequential(target())
    sugar.apply_sugar(model, "noise_relu", alpha=std, target=target)
    torch.manual_seed(71)
    expected = torch.where(x > 0, torch.ones_like(x), torch.randn_like(x) * std)
    if target is nn.ReLU6:
        expected[x >= 6] = 0
    torch.manual_seed(71)
    y = model(x)
    assert torch.equal(y, target()(x))
    weights = torch.arange(1., 8.)
    (y * weights).sum().backward()
    assert torch.equal(x.grad, expected * weights)
    if std > 0:
        assert torch.all(x.grad[x <= 0] != 0)


def test_noise_resamples_between_forwards():
    model = nn.Sequential(nn.ReLU())
    sugar.apply_sugar(model, "noise_relu", alpha=0.1)
    gradients = []
    torch.manual_seed(72)
    for _ in range(2):
        x = -torch.ones(20, requires_grad=True)
        gradients.append(torch.autograd.grad(model(x).sum(), x)[0])
    assert not torch.equal(*gradients)
