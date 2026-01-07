from codegen_backend.groups.builtin.conv.group import ConvGroup
from codegen_backend.groups.builtin.elementwise.group import ElementwiseGroup
from codegen_backend.groups.builtin.embedding.group import EmbeddingGroup
from codegen_backend.groups.builtin.pooling.group import PoolingGroup
from codegen_backend.groups.builtin.reductions.group import ReductionsGroup
from codegen_backend.groups.builtin.tensor.group import TensorGroup
from codegen_backend.groups.registry import register_group


def register_builtin_groups() -> None:
    register_group(ElementwiseGroup())
    register_group(ReductionsGroup())
    register_group(PoolingGroup())
    register_group(ConvGroup())
    register_group(EmbeddingGroup())
    register_group(TensorGroup())


__all__ = ["register_builtin_groups"]
