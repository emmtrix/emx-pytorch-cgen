from __future__ import annotations

from dataclasses import dataclass

import codegen_backend.groups.registry as registry_module
from codegen_backend.groups.base import OperatorGroupDefinition
from codegen_backend.groups.registry import get_group_registry, register_group


@dataclass(frozen=True)
class DummyGroup(OperatorGroupDefinition):
    name: str = "dummy"

    def build_supported_ops(self):
        return {}

    def build_target_registry(self, supported_ops):
        return {}

    def build_analyzers(self, supported_ops, target_registry):
        return ()


def test_register_group_adds_custom_group(monkeypatch):
    monkeypatch.setattr(registry_module, "_REGISTERED_GROUPS", {})
    monkeypatch.setattr(registry_module, "_GROUP_REGISTRY", None)
    monkeypatch.setattr(registry_module, "_DEFAULT_GROUPS_LOADED", True)
    monkeypatch.setattr(registry_module, "_ENTRY_POINTS_LOADED", True)

    register_group(DummyGroup())

    registry = get_group_registry()
    group_names = [group.name for group in registry.groups]

    assert "dummy" in group_names
