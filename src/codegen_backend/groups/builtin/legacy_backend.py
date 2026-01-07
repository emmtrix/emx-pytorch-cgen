from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping

from codegen_backend.kinds import OpKindHandlerFactory
from codegen_backend.ops_registry_conv import build_supported_ops as build_conv_ops
from codegen_backend.ops_registry_elementwise import (
    build_supported_ops as build_elementwise_ops,
)
from codegen_backend.ops_registry_embedding import (
    build_supported_ops as build_embedding_ops,
)
from codegen_backend.ops_registry_pooling import build_supported_ops as build_pooling_ops
from codegen_backend.ops_registry_reductions import (
    build_supported_ops as build_reductions_ops,
)
from codegen_backend.ops_registry_tensor import build_supported_ops as build_tensor_ops
from codegen_backend.registry import _TargetInfo, build_target_registry
from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class BaseBackendGroup:
    name: str = "base"

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
        return []

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        return {}

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return {}


@dataclass(frozen=True)
class LegacyBackendGroup:
    name: str = "legacy"

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
        from codegen_backend.groups.builtin.conv import handlers as conv_handlers
        from codegen_backend.groups.builtin.elementwise import (
            handlers as elementwise_handlers,
        )
        from codegen_backend.groups.builtin.embedding import (
            handlers as embedding_handlers,
        )
        from codegen_backend.groups.builtin.pooling import handlers as pooling_handlers
        from codegen_backend.groups.builtin.reductions import (
            handlers as reductions_handlers,
        )
        from codegen_backend.groups.builtin.tensor import handlers as tensor_handlers

        return [
            elementwise_handlers.ElementwiseKindHandlerFactory(),
            reductions_handlers.ReductionsKindHandlerFactory(),
            pooling_handlers.PoolingKindHandlerFactory(),
            conv_handlers.ConvKindHandlerFactory(),
            embedding_handlers.EmbeddingKindHandlerFactory(),
            tensor_handlers.TensorKindHandlerFactory(),
        ]

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        supported_ops: Dict[str, _OpSpec] = {}
        supported_ops.update(build_elementwise_ops())
        supported_ops.update(build_reductions_ops())
        supported_ops.update(build_pooling_ops())
        supported_ops.update(build_conv_ops())
        supported_ops.update(build_embedding_ops())
        supported_ops.update(build_tensor_ops())
        return supported_ops

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return build_target_registry(self.supported_ops())


OperatorGroup = LegacyBackendGroup


__all__ = ["BaseBackendGroup", "LegacyBackendGroup", "OperatorGroup"]
