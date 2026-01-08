from codegen_backend.groups.builtin.conv.group import ConvGroup
from codegen_backend.groups.builtin.elementwise.group import ElementwiseGroup
from codegen_backend.groups.builtin.embedding.group import EmbeddingGroup
from codegen_backend.groups.builtin.pooling.group import PoolingGroup
from codegen_backend.groups.builtin.reductions.group import ReductionsGroup
from codegen_backend.groups.builtin.tensor.group import TensorGroup
from codegen_backend.groups.registry import GroupRegistryBuilder


def register_builtin_groups(builder: GroupRegistryBuilder) -> None:
    builder.register_group(ElementwiseGroup())
    builder.register_group(ReductionsGroup())
    builder.register_group(PoolingGroup())
    builder.register_group(ConvGroup())
    builder.register_group(EmbeddingGroup())
    builder.register_group(TensorGroup())


__all__ = ["register_builtin_groups"]
