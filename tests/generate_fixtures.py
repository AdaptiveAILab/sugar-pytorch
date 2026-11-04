"""Regenerate the golden fixtures for the permanent parity tests.

test_fixtures.py runs this automatically whenever the fixtures are missing
or were recorded under a different torch build; set SUGAR_NO_FIXTURE_REGEN=1
to disable that.  Manual use (from the sugar-pytorch directory):

    python tests/generate_fixtures.py

The recorded behaviour comes from the reference copies of the
original experiment code in tests/_reference.py, applied to the model
architectures vendored in tests/_models.py (verbatim copies of the
experiment model definitions); the mobilenet nelu entry is recorded from
the fixed package SUGAR_NeLU (F.relu6 forward and zero gradient at x >= 6 -- the original uncapped
F.relu forward was a bug).  test_fixtures.py then asserts the `sugar`
package reproduces the recordings exactly.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TESTS_DIR = Path(__file__).resolve().parent

import _models  # noqa: E402
import _reference as ref  # noqa: E402
import sugar  # noqa: E402
from torchvision.models import mobilenet_v2  # noqa: E402


def build_vgg16():
    torch.manual_seed(0)
    return _models.VGG16(num_classes=100).eval()


def build_resnet18():
    torch.manual_seed(0)
    return _models.resnet18_cifar(num_classes=100).eval()


def build_mobilenet_v2():
    torch.manual_seed(0)
    return mobilenet_v2(weights=None).eval()


def build_toy_net():
    torch.manual_seed(4)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16),
                         nn.ReLU(), nn.Linear(16, 4))


def _tree(model):
    # reference classes carry prefixes; record the original (package) names
    return [(name, ref.REF_TO_PKG_CLASS.get(type(mod).__name__, type(mod).__name__))
            for name, mod in model.named_modules()]


def _record(model, x, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    h = x.detach().clone().requires_grad_(True)
    y = model(h)
    y.sum().backward()
    return {"out": y.detach().clone(), "grad": h.grad.detach().clone()}


def main():
    torch.manual_seed(2)
    x32 = torch.randn(2, 3, 32, 32)
    torch.manual_seed(3)
    x8 = torch.randn(5, 8)

    data = {"meta": {
        "torch_version": torch.__version__,
        "schema": 3,
        "source": "tests/_reference.py (reference experiment code) + "
                  "tests/_models.py (vendored experiment model definitions); "
                  "mobilenet nelu from the fixed package SUGAR_NeLU (F.relu6 forward and zero gradient at x >= 6)",
    }}

    # ---------------- VGG16 (vendored experiment architecture) ----------------
    data["vgg16"] = {
        "state_dict_keys": list(build_vgg16().state_dict().keys()),
        "relu_count": sum(isinstance(m, nn.ReLU) for m in build_vgg16().modules()),
        "input": x32,
        "fgi": {},
        "baseline": {},
        "swapback": {},
    }
    for fgi_type, alpha in (("bsilu", 0.1), ("nelu", 0.01), ("noise_relu", 0.01)):
        model = build_vgg16()
        ref.cnn_perform_relu_fgi(model, fgi_type, alpha=alpha, compile_layer=False)
        rec = {"tree": _tree(model), "layer_type": "ReluSurrogate"}
        rec.update(_record(model, x32, seed=77 if fgi_type == "noise_relu" else None))
        data["vgg16"]["fgi"][fgi_type] = rec

    model = build_vgg16()
    _, layer_type = ref.cnn_perform_relu_fgi(model, "bsilu", compile_layer=True)
    data["vgg16"]["fgi"]["bsilu_compiled"] = {
        "tree": _tree(model), "layer_type": layer_type.__name__,
    }

    # non-FGI baseline path (gelu)
    Surrogate = ref.make_cnn_train_surrogate(F.gelu)
    model = build_vgg16()
    ref.cnn_train_replace_layers(model, nn.ReLU, lambda: Surrogate())
    rec = {"tree": _tree(model), "layer_type": "Surrogate"}
    rec.update(_record(model, x32))
    data["vgg16"]["baseline"]["gelu"] = rec

    # eval-time swap-back (FGI bsilu model -> plain leaky activation)
    model = build_vgg16()
    model, layer_type = ref.cnn_perform_relu_fgi(model, "bsilu",
                                                 compile_layer=False)
    Surrogate = ref.make_cnn_train_surrogate(F.leaky_relu)
    ref.cnn_train_replace_layers(model, layer_type, lambda: Surrogate())
    rec = {"tree": _tree(model), "layer_type": "Surrogate"}
    rec.update(_record(model, x32))
    data["vgg16"]["swapback"]["leaky"] = rec

    # ---------------- resnet18 (vendored experiment architecture) ----------------
    data["resnet18"] = {
        "state_dict_keys": list(build_resnet18().state_dict().keys()),
        "relu_count": sum(isinstance(m, nn.ReLU) for m in build_resnet18().modules()),
        "input": x32,
        "fgi": {},
    }
    model = build_resnet18()
    ref.cnn_perform_relu_fgi(model, "bsilu", compile_layer=False)
    rec = {"tree": _tree(model), "layer_type": "ReluSurrogate"}
    rec.update(_record(model, x32))
    data["resnet18"]["fgi"]["bsilu"] = rec

    # ---------------- mobilenet_v2 ----------------
    data["mobilenet_v2"] = {
        "state_dict_keys": list(build_mobilenet_v2().state_dict().keys()),
        "relu6_count": sum(isinstance(m, nn.ReLU6) for m in build_mobilenet_v2().modules()),
        "input": x32,
        "variants": {},
    }
    # nelu: the fixed package behaviour (F.relu6 forward and zero gradient at x >= 6)
    model = build_mobilenet_v2()
    count = sugar.apply_sugar(model, "nelu", alpha=0.05, target=nn.ReLU6)
    rec = {"count": count, "tree": _tree(model),
           "list_activations": sorted(sugar.list_activations(model))}
    rec.update(_record(model, x32))
    data["mobilenet_v2"]["variants"]["nelu"] = rec

    # bsilu / leaky: the original code (verbatim reference classes)
    for variant, cls in (("bsilu", ref.MobilenetSUGAR_BSiLU),
                         ("leaky", ref.MobilenetSUGAR_LeakyReLU)):
        def ctor(cls=cls, variant=variant):
            return cls() if variant == "bsilu" else cls(0.01)

        model = build_mobilenet_v2()
        count = ref.mobilenet_replace_modules_(model, nn.ReLU6, ctor)
        rec = {"count": count, "tree": _tree(model),
               "list_activations": sorted(ref.REF_TO_PKG_CLASS.get(name, name)
                                          for name in ref.mobilenet_list_activations(model))}
        rec.update(_record(model, x32))
        data["mobilenet_v2"]["variants"][variant] = rec

    # ---------------- toy ----------------
    data["toy"] = {"input": x8, "types": {}}
    for fgi in ref.TOY_TYPES:
        model = build_toy_net()
        model, layer_type = ref.toy_perform_relu_fgi(model, fgi)
        rec = {"tree": _tree(model),
               "layer_type": ref.REF_TO_PKG_CLASS.get(layer_type.__name__,
                                                     layer_type.__name__)}
        rec.update(_record(model, x8))
        data["toy"]["types"][fgi] = rec

    # ---------------- noise_relu RNG stream (VGG16, vendored) ----------------
    model = build_vgg16()
    ref.cnn_perform_relu_fgi(model, "noise_relu", alpha=0.01, compile_layer=False)
    torch.manual_seed(99)
    data["noise_stream"] = {"outputs": [model(x32).detach().clone() for _ in range(3)]}

    out_dir = TESTS_DIR / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / "golden.pt")
    print(f"Wrote {out_dir / 'golden.pt'} (torch {torch.__version__})")


if __name__ == "__main__":
    main()
