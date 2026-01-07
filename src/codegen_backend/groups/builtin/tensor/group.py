from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping

from codegen_backend.groups.builtin.tensor import handlers
from codegen_backend.kinds import OpKindHandlerFactory
from codegen_backend.ops_registry_tensor import build_supported_ops
from codegen_backend.registry import _TargetInfo, build_target_registry
from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class TensorGroup:
    name: str = "tensor"

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
        return [handlers.TensorKindHandlerFactory()]

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        return build_supported_ops()

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return build_target_registry(self.supported_ops())


__all__ = ["TensorGroup"]
