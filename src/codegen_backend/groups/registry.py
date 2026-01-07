from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Protocol

from codegen_backend.kinds import HandlerContext, OpKindHandler
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import OpKind, _OpSpec


class Group(Protocol):
    name: str

    def kind_handlers(
        self, context: HandlerContext
    ) -> Dict[OpKind, OpKindHandler]: ...

    @property
    def supported_ops(self) -> Mapping[str, _OpSpec]: ...

    @property
    def target_registry(self) -> Mapping[object, _TargetInfo]: ...


@dataclass
class GroupRegistry:
    groups: list[Group] = field(default_factory=list)

    def register(self, group: Group) -> None:
        self.groups.append(group)

    def build_kind_handlers(
        self, context: HandlerContext
    ) -> Dict[OpKind, OpKindHandler]:
        merged: Dict[OpKind, OpKindHandler] = {}
        for group in self.groups:
            merged.update(group.kind_handlers(context))
        return merged

    def _merge_overlays(
        self, overlays: Iterable[Mapping[object, object]]
    ) -> Dict[object, object]:
        merged: Dict[object, object] = {}
        for overlay in overlays:
            merged.update(overlay)
        return merged

    def build_supported_ops(self) -> Dict[str, _OpSpec]:
        overlays = [group.supported_ops for group in self.groups]
        return {key: value for key, value in self._merge_overlays(overlays).items()}

    def build_target_registry(self) -> Dict[object, _TargetInfo]:
        overlays = [group.target_registry for group in self.groups]
        return {key: value for key, value in self._merge_overlays(overlays).items()}


_GROUP_REGISTRY: GroupRegistry | None = None


def get_group_registry() -> GroupRegistry:
    global _GROUP_REGISTRY
    if _GROUP_REGISTRY is None:
        from codegen_backend.groups.legacy import LegacyGroup

        registry = GroupRegistry()
        registry.register(LegacyGroup())
        _GROUP_REGISTRY = registry
    return _GROUP_REGISTRY


__all__ = ["Group", "GroupRegistry", "get_group_registry"]
