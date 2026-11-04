"""Shared pytest configuration for the sugar-pytorch test suite.

The real-architecture models used by the golden parity net
(tests/test_fixtures.py) come from tests/_models.py: vendored verbatim copies
of the experiment model definitions, so the suite runs without the
experiments tree.
"""

import functools

import pytest
import torch
import torch.nn as nn

import _models


def _dynamo_ok():
    """Probe once whether torch.compile runs here (eager backend needs no C++ compiler)."""
    try:
        compiled = torch.compile(nn.ReLU(), backend="eager")
        with torch.no_grad():
            compiled(torch.randn(3))
        return True
    except Exception:
        return False


DYNAMO_OK = _dynamo_ok()

requires_dynamo = pytest.mark.skipif(
    not DYNAMO_OK, reason="torch.compile(backend='eager') is not functional in this environment"
)


@pytest.fixture
def compile_eager(monkeypatch):
    """Make torch.compile use backend='eager' so compiled layers can execute
    without a C++ toolchain (structure/wrapping semantics unchanged)."""
    eager_compile = functools.partial(torch.compile, backend="eager")
    monkeypatch.setattr(torch, "compile", eager_compile)
    return eager_compile


# ---------------------------------------------------------------------------
# Deterministic model builders (identical twins via fixed seeds)
# ---------------------------------------------------------------------------

def build_relu_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8),
                         nn.ReLU(), nn.Linear(8, 3))


def build_relu6_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU6(), nn.Linear(8, 8),
                         nn.ReLU6(), nn.Linear(8, 3))


def build_vgg16():
    torch.manual_seed(0)
    return _models.VGG16(num_classes=100).eval()


def build_resnet18():
    try:
        torch.manual_seed(0)
        return _models.resnet18_cifar(num_classes=100).eval()
    except ImportError:
        pytest.skip("torchvision not available")


def build_mobilenet_v2():
    try:
        from torchvision.models import mobilenet_v2 as _mv2
    except Exception:
        pytest.skip("torchvision not available")
    torch.manual_seed(0)
    return _mv2(weights=None).eval()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def x_act():
    """1-D inputs incl. the boundary values 0 and 6."""
    torch.manual_seed(1)
    special = torch.tensor([0.0, -0.0, 6.0, -6.0, 5.9999, 6.0001, 1e-8, -1e-8])
    return torch.cat([torch.randn(64) * 3.0, special])


@pytest.fixture
def img_32():
    torch.manual_seed(2)
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def small_relu_net():
    return build_relu_net()


@pytest.fixture
def small_relu6_net():
    return build_relu6_net()


@pytest.fixture
def vgg16():
    return build_vgg16()


@pytest.fixture
def resnet18_cifar():
    return build_resnet18()


@pytest.fixture
def mobilenet_v2():
    return build_mobilenet_v2()
