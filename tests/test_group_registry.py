from __future__ import annotations

from dataclasses import dataclass

from codegen_backend.groups.base import OperatorGroupDefinition
from codegen_backend.groups.registry import GroupRegistryBuilder


@dataclass(frozen=True)
class DummyGroup(OperatorGroupDefinition):
    name: str = "dummy"

    def build_supported_ops(self):
        return {}

    def build_target_registry(self, supported_ops):
        return {}

    def build_analyzers(self, supported_ops, target_registry):
        return ()


def test_register_group_adds_custom_group():
    builder = GroupRegistryBuilder()
    builder.register_group(DummyGroup())
    registry = builder.build()
    group_names = [group.name for group in registry.groups]

    assert "dummy" in group_names
