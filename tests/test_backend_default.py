from __future__ import annotations

from dataclasses import dataclass

import codegen_backend.backend as backend_module
import codegen_backend.groups.registry as registry_module
from codegen_backend.groups.base import OperatorGroupDefinition
from codegen_backend.groups.registry import register_group
from codegen_backend.specs import OpKind, _OpSpec


@dataclass(frozen=True)
class DummyGroup(OperatorGroupDefinition):
    name: str = "dummy_group"

    def build_supported_ops(self):
        return {
            "dummy_op": _OpSpec(
                name="dummy_op",
                kind=OpKind.UNARY,
                symbol=None,
                supported_targets=set(),
            )
        }

    def build_target_registry(self, supported_ops):
        return {}

    def build_analyzers(self, supported_ops, target_registry):
        return ()


def test_default_backend_refreshes_after_register_group(monkeypatch):
    monkeypatch.setattr(registry_module, "_REGISTERED_GROUPS", {})
    monkeypatch.setattr(registry_module, "_GROUP_REGISTRY", None)
    monkeypatch.setattr(registry_module, "_DEFAULT_GROUPS_LOADED", True)
    monkeypatch.setattr(registry_module, "_ENTRY_POINTS_LOADED", True)
    monkeypatch.setattr(backend_module, "_DEFAULT_BACKEND", None)
    monkeypatch.setattr(backend_module, "_DEFAULT_BACKEND_GROUP_REGISTRY", None)

    backend_before = backend_module.get_default_backend()
    assert "dummy_op" not in backend_before.supported_ops

    register_group(DummyGroup())

    backend_after = backend_module.get_default_backend()
    assert "dummy_op" in backend_after.supported_ops
    assert backend_after is not backend_before
