from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list

from codegen_backend.backend_helpers import _resolve_alias
from codegen_backend.compile import compile_or_load
from codegen_backend.dtypes import _CodegenDType, _EMBEDDING_INDEX_DTYPES
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _GenericGraph, _GenericLibrary
from codegen_backend.kinds import OpKind
from codegen_backend.graph_builder import GraphBuilder
from codegen_backend.emitter import Emitter


_C_SRC_DIR = Path(__file__).resolve().parents[2] / "csrc"


class Compiler:
    def __init__(self, builder: GraphBuilder, emitter: Emitter) -> None:
        self._builder = builder
        self._emitter = emitter

    def get_source(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> str:
        graph = self._builder.build(gm, example_inputs)
        return self._emitter.emit(graph)

    def compile_library(self, graph: _GenericGraph) -> _GenericLibrary:
        return self._compile_generic_library(graph)

    def compile_graph(
        self, gm: torch.fx.GraphModule, example_inputs: List[object]
    ) -> Callable[..., torch.Tensor]:
        return self._compile_graph(gm, example_inputs)

    def _compile_generic_library(self, graph: _GenericGraph) -> _GenericLibrary:
        source = self._emitter.emit(graph)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        entry_name = f"ref_codegen_main_{graph.dtype.suffix}"
        input_shapes = tuple(graph.shapes[node] for node in graph.tensor_placeholders)
        input_strides = tuple(graph.strides[node] for node in graph.tensor_placeholders)
        return compile_or_load(
            source,
            digest,
            entry_name=entry_name,
            include_dirs=[_C_SRC_DIR],
            input_shapes=input_shapes,
            input_strides=input_strides,
            output_shape=graph.shapes[graph.output_value],
            dtype=graph.dtype,
        )

    def _validate_runtime_inputs(
        self,
        inputs: Iterable[torch.Tensor],
        expected_dtypes: Sequence[torch.dtype],
        graph_dtype: _CodegenDType,
    ) -> None:
        for tensor, expected_dtype in zip(inputs, expected_dtypes):
            if expected_dtype is torch.bool:
                if tensor.dtype is not torch.bool:
                    raise CodegenBackendError(
                        "codegen backend expects boolean condition tensors"
                    )
            elif expected_dtype in _EMBEDDING_INDEX_DTYPES:
                if tensor.dtype is not expected_dtype:
                    raise CodegenBackendError(
                        "codegen backend expects int32 or int64 index tensors"
                    )
            elif tensor.dtype is not graph_dtype.torch_dtype:
                raise CodegenBackendError(
                    f"codegen backend supports only {graph_dtype.torch_dtype} tensors"
                )
            if tensor.device.type != "cpu":
                raise CodegenBackendError("codegen backend supports only CPU tensors")

    def _compile_graph(
        self, gm: torch.fx.GraphModule, example_inputs: List[object]
    ) -> Callable[..., torch.Tensor]:
        graph = self._builder.build(gm, example_inputs)
        conv_contiguous_indices = tuple(
            sorted(
                {
                    graph.tensor_placeholders.index(input_node)
                    for op_node in graph.op_nodes
                    if op_node.spec.kind in {OpKind.CONV1D, OpKind.CONV2D}
                    for input_node in op_node.inputs
                    if input_node in graph.tensor_placeholders
                }
            )
        )

        def _normalize_conv_inputs(
            inputs: Sequence[object],
        ) -> List[object]:
            normalized = list(inputs)
            for index in conv_contiguous_indices:
                placeholder = graph.tensor_placeholders[index]
                placeholder_index = graph.placeholders.index(placeholder)
                value = normalized[placeholder_index]
                if isinstance(value, torch.Tensor) and not value.is_contiguous():
                    normalized[placeholder_index] = value.contiguous()
            return normalized

        normalized_example_inputs = (
            _normalize_conv_inputs(example_inputs)
            if conv_contiguous_indices
            else list(example_inputs)
        )
        graph = self._builder.build(gm, normalized_example_inputs)
        lib = self._compile_generic_library(graph)
        output_structure = graph.output_structure
        output_value = graph.output_value
        output_inplace_input = graph.output_inplace_input
        library_cache: Dict[
            Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]],
            _GenericLibrary,
        ] = {
            (lib.input_shapes, lib.input_strides): lib,
        }

        def _recompile(new_inputs: Sequence[object]) -> None:
            nonlocal graph, lib, output_inplace_input
            graph = self._builder.build(gm, _normalize_conv_inputs(new_inputs))
            lib = self._compile_generic_library(graph)
            output_inplace_input = graph.output_inplace_input

        def resolve_output(value: object, env: Dict[torch.fx.Node, object]) -> object:
            if isinstance(value, torch.fx.Node):
                return env[value]
            if isinstance(value, (list, tuple, immutable_list)):
                resolved = [resolve_output(item, env) for item in value]
                return type(value)(resolved)
            return value

        def compiled(*args: object, **kwargs: object) -> object:
            if kwargs:
                placeholder_targets = [node.target for node in graph.placeholders]
                normalized_args = list(args)
                for name in placeholder_targets[len(normalized_args) :]:
                    if name in kwargs:
                        normalized_args.append(kwargs[name])
            else:
                normalized_args = list(args)
            if len(normalized_args) != len(graph.placeholders):
                raise CodegenBackendError(
                    f"codegen backend expects {len(graph.placeholders)} inputs, got {len(normalized_args)}"
                )
            env: Dict[torch.fx.Node, object] = {}
            input_tensors = []
            for node, value in zip(graph.placeholders, normalized_args):
                env[node] = value
                if node in graph.tensor_placeholders:
                    if not isinstance(value, torch.Tensor):
                        raise CodegenBackendError(
                            "codegen backend expects tensor inputs only"
                        )
                    input_tensors.append(value)
            expected_dtypes = [
                graph.dtypes[node] for node in graph.tensor_placeholders
            ]
            self._validate_runtime_inputs(
                input_tensors, expected_dtypes, graph.dtype
            )

            contiguous_inputs = list(input_tensors)
            if conv_contiguous_indices:
                for index in conv_contiguous_indices:
                    if not contiguous_inputs[index].is_contiguous():
                        contiguous_inputs[index] = contiguous_inputs[
                            index
                        ].contiguous()

            input_shapes = tuple(tuple(tensor.shape) for tensor in contiguous_inputs)
            input_strides = tuple(tuple(tensor.stride()) for tensor in contiguous_inputs)
            cache_key = (input_shapes, input_strides)
            cached_lib = library_cache.get(cache_key)
            if cached_lib is None:
                analysis_inputs = list(normalized_args)
                if conv_contiguous_indices:
                    for index in conv_contiguous_indices:
                        placeholder = graph.tensor_placeholders[index]
                        placeholder_index = graph.placeholders.index(placeholder)
                        analysis_inputs[placeholder_index] = contiguous_inputs[
                            index
                        ]
                updated_graph = self._builder.build(gm, analysis_inputs)
                cached_lib = self._compile_generic_library(updated_graph)
                library_cache[cache_key] = cached_lib
            lib = cached_lib
            if output_inplace_input is not None:
                original_input = env[output_inplace_input]
                if not isinstance(original_input, torch.Tensor):
                    raise CodegenBackendError(
                        "codegen backend expects tensor inputs only"
                    )
                inplace_index = graph.tensor_placeholders.index(output_inplace_input)
                inplace_out = contiguous_inputs[inplace_index]
                lib.run(contiguous_inputs, inplace_out)
                if inplace_out is not original_input:
                    original_input.copy_(inplace_out)
                env[output_value] = original_input
            else:
                output_dtype = graph.dtypes[output_value]
                device = (
                    contiguous_inputs[0].device
                    if contiguous_inputs
                    else torch.device("cpu")
                )
                if graph.output_op.spec.kind == OpKind.EMPTY_STRIDED:
                    out = torch.empty_strided(
                        graph.shapes[output_value],
                        graph.strides[output_value],
                        dtype=output_dtype,
                        device=device,
                    )
                else:
                    out = torch.empty(
                        lib.output_shape,
                        dtype=output_dtype,
                        device=device,
                    )
                lib.run(contiguous_inputs, out)
                env[output_value] = out
            if graph.alias_map:
                for alias, source in graph.alias_map.items():
                    resolved = _resolve_alias(source, graph.alias_map)
                    if resolved in env:
                        env[alias] = env[resolved]
            if graph.empty_outputs:
                device = (
                    contiguous_inputs[0].device
                    if contiguous_inputs
                    else torch.device("cpu")
                )
                for node in graph.empty_outputs:
                    if node not in env:
                        env[node] = torch.empty(
                            graph.shapes[node],
                            dtype=graph.dtypes[node],
                            device=device,
                        )
            return resolve_output(output_structure, env)

        return compiled
