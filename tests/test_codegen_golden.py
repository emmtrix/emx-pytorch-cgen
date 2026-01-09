import os
from pathlib import Path

import pytest
import torch
from codegen_backend import CodegenBackend
from codegen_backend.errors import CodegenBackendError
from codegen_backend.indexing import _emit_strided_access, _format_strided_access

REFERENCE_DIR = Path(__file__).resolve().parent / "codegen_refs"
BACKEND = CodegenBackend()


def _assert_codegen_source_matches(
    reference_name: str, source_fn, fn, example_inputs, reference_dir: Path | None = None
) -> None:
    reference_root = reference_dir or REFERENCE_DIR
    reference_path = reference_root / reference_name
    gm = torch.fx.symbolic_trace(fn)
    source = source_fn(gm, example_inputs).lstrip()
    if os.getenv("UPDATE_REFS"):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(source, encoding="utf-8")
    expected = reference_path.read_text(encoding="utf-8")
    assert source == expected


def _assert_codegen_graph_matches(
    reference_name: str,
    source_fn,
    gm: torch.fx.GraphModule,
    example_inputs,
    reference_dir: Path | None = None,
) -> None:
    reference_root = reference_dir or REFERENCE_DIR
    reference_path = reference_root / reference_name
    source = source_fn(gm, example_inputs).lstrip()
    if os.getenv("UPDATE_REFS"):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(source, encoding="utf-8")
    expected = reference_path.read_text(encoding="utf-8")
    assert source == expected


def add_chain_fn(a, b, c):
    return (a + b) + c


def add_broadcast_fn(a, b):
    return a + b


def add_strided_fn(a, b):
    return a + b


def sub_chain_fn(a, b, c):
    return (a - b) - c


def mul_chain_fn(a, b, c):
    return (a * b) * c


def mixed_ops_fn(a, b, c):
    return torch.relu(a + b) - c


def atan_fn(a):
    return torch.atan(a)


def mish_fn(a):
    return torch.ops.aten.mish.default(a)


def inplace_fn(a):
    b = torch.atan(a)
    b = torch.ops.aten.add_.Tensor(b, a)
    return torch.mul(b, a)


def reduction_global_fn(a):
    return a.sum()


def reduction_mean_global_fn(a):
    return a.mean()


def reduction_mean_dim_fn(a):
    return a.mean(dim=1)


def reduction_dim_fn(a):
    return a.sum(dim=1)


def reduction_keepdim_fn(a):
    return a.sum(dim=1, keepdim=True)


def reduction_strided_fn(a):
    return a.sum(dim=0)


def reduction_broadcast_fn(a, b):
    return (a + b).sum(dim=1)


def argmax_dim_fn(a):
    return torch.ops.aten.argmax.default(a, 1, False)


def where_fn(condition, a, b):
    return torch.where(condition, a, b)


def cat_fn(a, b):
    return torch.cat([a, b], dim=1)


def conv2d_fn(a, weight, bias):
    return torch.ops.aten.conv2d.default(
        a, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
    )


def max_pool2d_fn(a):
    return torch.ops.aten.max_pool2d.default(
        a, (2, 2), (1, 1), (0, 0), (1, 1), False
    )


def temp_alloc_fn(a, b, c, d):
    return (a + b) + (c + d)


def fill_full_like_fn(a):
    return torch.ops.aten.full_like.default(a, 3.0)


def view_reshape_fn(a):
    return torch.ops.aten.reshape.default(a, (2, 12))


def flip_fn(a):
    return torch.ops.aten.flip.default(a, [1])


def repeat_fn(a):
    return torch.ops.aten.repeat.default(a, [1, 2])


def arange_fn(start, end, step):
    return torch.ops.aten.arange.start_step(start, end, step)


def softmax_fn(a):
    return torch.softmax(a, 1)


def cumsum_fn(a):
    return torch.ops.aten.cumsum.default(a, 1)


def diagonal_fn(a):
    return torch.ops.aten.diagonal.default(a, 0, 0, 1)


def addmm_fn(a, b, c):
    return torch.ops.aten.addmm.default(a, b, c)


def addbmm_fn(a, b, c):
    return torch.ops.aten.addbmm.default(a, b, c)


def addmv_fn(a, b, c):
    return torch.ops.aten.addmv.default(a, b, c)


def addr_fn(a, b, c):
    return torch.ops.aten.addr.default(a, b, c)


def matmul_fn(a, b):
    return torch.ops.aten.matmul.default(a, b)


def linear_fn(a, weight, bias):
    return torch.ops.aten.linear.default(a, weight, bias)


def conv1d_fn(a, weight, bias):
    return torch.ops.aten.conv1d.default(
        a, weight, bias, stride=(1,), padding=(0,), dilation=(1,), groups=1
    )


def max_pool1d_fn(a):
    return torch.ops.aten.max_pool1d.default(a, (2,), (1,), (0,), (1,), False)


def max_pool3d_fn(a):
    output, _ = torch.ops.aten.max_pool3d_with_indices.default(
        a, (2, 2, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1), False
    )
    return output


def max_pool2d_backward_fn(a, grad):
    return torch.ops.aten._adaptive_avg_pool2d_backward.default(grad, a)


def embedding_fn(weight, indices):
    return torch.ops.aten.embedding.default(weight, indices, -1, False, False)


def embedding_bag_fn(weight, indices, offsets):
    return torch.ops.aten._embedding_bag.default(
        weight, indices, offsets, False, 0, False, None, False, -1
    )[0]


def embedding_dense_backward_fn(grad_output, indices):
    return torch.ops.aten.embedding_dense_backward.default(
        grad_output, indices, 4, -1, False
    )


def gather_fn(a, index):
    return torch.ops.aten.gather.default(a, 1, index)


def index_put_fn(a, index, values):
    return torch.ops.aten.index_put.default(a, (index,), values, False)


def index_select_fn(a, index):
    return torch.ops.aten.index_select.default(a, 0, index)


def batch_norm_fn(a, weight, bias, running_mean, running_var):
    return torch.ops.aten._native_batch_norm_legit.default(
        a, weight, bias, running_mean, running_var, False, 0.1, 1e-5
    )


def layer_norm_fn(a, weight, bias):
    return torch.ops.aten.native_layer_norm.default(a, [4], weight, bias, 1e-5)


def layer_norm_backward_fn(a, weight, bias, mean, rstd, grad_out):
    return torch.ops.aten.native_layer_norm_backward.default(
        grad_out, a, [4], mean, rstd, weight, bias, [True, True, True]
    )


def group_norm_fn(a, weight, bias):
    return torch.ops.aten.native_group_norm.default(
        a, weight, bias, 2, 2, 4, 1, 1e-5
    )


def group_norm_backward_fn(a, weight, mean, rstd, grad_out):
    return torch.ops.aten.native_group_norm_backward.default(
        grad_out, a, mean, rstd, weight, 2, 2, 4, 1, [True, True, True]
    )


def pdist_fn(a):
    return torch.ops.aten._pdist_forward.default(a, 2.0)


def cdist_fn(a, b):
    return torch.ops.aten._cdist_forward.default(a, b, 2.0, None)


def pad_fn(a):
    return torch.ops.aten.constant_pad_nd.default(a, [1, 1, 1, 1], 0.0)


def scalar_tensor_fn(value):
    return torch.ops.aten.scalar_tensor.default(value, dtype=torch.float32)


def random_fn(size):
    return torch.ops.aten.rand.default(size)


def randperm_fn(n):
    return torch.ops.aten.randperm.default(n)


def resize_fn(a):
    return torch.ops.aten.resize_.default(a, [2, 3])


def col2im_fn(a):
    return torch.ops.aten.col2im.default(
        a, [2, 2], [1, 1], [1, 1], [0, 0], [1, 1]
    )


def masked_scatter_fn(a, mask, values):
    return torch.ops.aten.masked_scatter.default(a, mask, values)


def select_scatter_fn(a, src):
    return torch.ops.aten.select_scatter.default(a, src, 0, 1)


def scatter_fn(a, index, src):
    return torch.ops.aten.scatter.src(a, 1, index, src)


def split_with_sizes_fn(a):
    return torch.ops.aten.split_with_sizes.default(a, [1, 2], 1)[0]


def nonzero_fn(a):
    return torch.ops.aten.nonzero.default(a)


def dropout_fn(a):
    return torch.ops.aten.native_dropout.default(a, 0.0, False)


def sort_fn(a):
    return torch.ops.aten.sort.default(a, 1, False)


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        (
            "binary_add_chain.c",
            add_chain_fn,
            BACKEND.get_generic_source,
            BACKEND.codegen_generic_backend,
        ),
        (
            "binary_sub_chain.c",
            sub_chain_fn,
            BACKEND.get_generic_source,
            BACKEND.codegen_generic_backend,
        ),
        (
            "binary_mul_chain.c",
            mul_chain_fn,
            BACKEND.get_generic_source,
            BACKEND.codegen_generic_backend,
        ),
    ],
)
def test_binary_handles_multiple_ops(
    reference_name, fn, source_fn, backend
):
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    c = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(reference_name, source_fn, fn, (a, b, c))
    compiled = torch.compile(fn, backend=backend)
    result = compiled(a, b, c)
    torch.testing.assert_close(result, fn(a, b, c))


def test_binary_handles_mixed_ops():
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    c = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "binary_mixed_ops.c", BACKEND.get_generic_source, mixed_ops_fn, (a, b, c)
    )
    compiled = torch.compile(mixed_ops_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, b, c)
    torch.testing.assert_close(result, mixed_ops_fn(a, b, c))


def test_binary_handles_add_broadcast():
    a = torch.randn(2, 1, 3, dtype=torch.float32)
    b = torch.randn(1, 4, 1, dtype=torch.float32)
    _assert_codegen_source_matches(
        "binary_add_broadcast.c", BACKEND.get_generic_source, add_broadcast_fn, (a, b)
    )
    compiled = torch.compile(add_broadcast_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, b)
    torch.testing.assert_close(result, add_broadcast_fn(a, b))


def test_binary_handles_strided_inputs():
    base = torch.randn(2, 3, dtype=torch.float32)
    a = base.t()
    b = torch.randn(2, 3, dtype=torch.float32).t()
    source = BACKEND.get_generic_source(torch.fx.symbolic_trace(add_strided_fn), (a, b))
    assert "((float*)a)" in source
    assert "((float*)b)" in source
    _assert_codegen_source_matches(
        "binary_add_strided.c", BACKEND.get_generic_source, add_strided_fn, (a, b)
    )
    compiled = torch.compile(add_strided_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, b)
    torch.testing.assert_close(result, add_strided_fn(a, b))


def test_conv2d_handles_conv2d():
    a = torch.randn(1, 2, 5, 5, dtype=torch.float32)
    weight = torch.randn(3, 2, 3, 3, dtype=torch.float32)
    bias = torch.randn(3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "conv2d_conv2d.c", BACKEND.get_generic_source, conv2d_fn, (a, weight, bias)
    )
    compiled = torch.compile(conv2d_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, weight, bias)
    torch.testing.assert_close(result, conv2d_fn(a, weight, bias))


def test_pool2d_handles_max_pool2d():
    a = torch.randn(1, 2, 4, 4, dtype=torch.float32)
    _assert_codegen_source_matches(
        "pool2d_max_pool2d.c", BACKEND.get_generic_source, max_pool2d_fn, (a,)
    )
    compiled = torch.compile(max_pool2d_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, max_pool2d_fn(a))


def test_unary_handles_atan():
    a = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "unary_atan.c", BACKEND.get_generic_source, atan_fn, (a,)
    )
    compiled = torch.compile(atan_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, atan_fn(a))


def test_unary_handles_mish():
    a = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "unary_mish.c", BACKEND.get_generic_source, mish_fn, (a,)
    )
    source = BACKEND.get_generic_source(torch.fx.symbolic_trace(mish_fn), (a,))
    assert "ref_scalar_f32_mish" in source
    compiled = torch.compile(mish_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, mish_fn(a))


def test_where_handles_where():
    condition = torch.tensor([[True, False, True]], dtype=torch.bool)
    a = torch.randn(1, 3, dtype=torch.float32)
    b = torch.randn(1, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "where_where.c", BACKEND.get_generic_source, where_fn, (condition, a, b)
    )
    compiled = torch.compile(where_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(condition, a, b)
    torch.testing.assert_close(result, where_fn(condition, a, b))


def test_concat_handles_cat():
    a = torch.randn(2, 2, dtype=torch.float32)
    b = torch.randn(2, 1, dtype=torch.float32)
    _assert_codegen_source_matches(
        "concat_cat.c", BACKEND.get_generic_source, cat_fn, (a, b)
    )
    compiled = torch.compile(cat_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, b)
    torch.testing.assert_close(result, cat_fn(a, b))


def test_binary_supports_inplace_ops():
    a = torch.randn(2, 3, dtype=torch.float32)
    expected = a.clone()
    gm = torch.fx.symbolic_trace(inplace_fn)
    source = BACKEND.get_generic_source(gm, (a,))
    assert "node2_add_f32(tmp_0, input_0, tmp_0);" in source
    _assert_codegen_source_matches(
        "binary_inplace_chain.c", BACKEND.get_generic_source, inplace_fn, (a,)
    )
    compiled = torch.compile(inplace_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    expected_result = inplace_fn(expected)
    torch.testing.assert_close(result, expected_result)
    torch.testing.assert_close(a, expected)


def test_binary_i32():
    a = torch.randint(0, 5, (2, 3), dtype=torch.int32)
    b = torch.randint(0, 5, (2, 3), dtype=torch.int32)
    _assert_codegen_source_matches(
        "binary_mul_chain_i32.c", BACKEND.get_generic_source, mul_chain_fn, (a, b, b)
    )
    compiled = torch.compile(mul_chain_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, b, b)
    torch.testing.assert_close(result, mul_chain_fn(a, b, b))


def test_binary_emit_strided_access_expressions():
    broadcast_expr = _format_strided_access(
        "b", (1, 3), (3, 1), (2, 3)
    )
    helper_expr = _emit_strided_access(
        "b", ("i0", "i1"), (3, 1), contig=False, sizes=(1, 3)
    )
    assert broadcast_expr == helper_expr
    assert broadcast_expr == "((float*)b)[i1 * 1]"
    assert (
        _emit_strided_access("a", ("i", "t"), (5, 1), contig=False, sizes=(2, 3))
        == "((float*)a)[i * 5 + t * 1]"
    )
    assert (
        _emit_strided_access("a", ("i", "t"), (5, 1), contig=True, sizes=(2, 3))
        == "a[i][t]"
    )


def test_reduction_handles_sum_global():
    a = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "reduction_sum_global.c",
        BACKEND.get_generic_source,
        reduction_global_fn,
        (a,),
    )
    compiled = torch.compile(reduction_global_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, reduction_global_fn(a))


def test_reduction_handles_mean_global():
    a = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "reduction_mean_global.c",
        BACKEND.get_generic_source,
        reduction_mean_global_fn,
        (a,),
    )
    compiled = torch.compile(
        reduction_mean_global_fn, backend=BACKEND.codegen_generic_backend
    )
    result = compiled(a)
    torch.testing.assert_close(result, reduction_mean_global_fn(a))


def test_reduction_handles_mean_dim():
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    _assert_codegen_source_matches(
        "reduction_mean_dim.c", BACKEND.get_generic_source, reduction_mean_dim_fn, (a,)
    )
    compiled = torch.compile(reduction_mean_dim_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, reduction_mean_dim_fn(a))


def test_reduction_handles_sum_dim():
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    _assert_codegen_source_matches(
        "reduction_sum_dim.c", BACKEND.get_generic_source, reduction_dim_fn, (a,)
    )
    compiled = torch.compile(reduction_dim_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, reduction_dim_fn(a))


def test_reduction_handles_sum_keepdim():
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    _assert_codegen_source_matches(
        "reduction_sum_keepdim.c",
        BACKEND.get_generic_source,
        reduction_keepdim_fn,
        (a,),
    )
    compiled = torch.compile(reduction_keepdim_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, reduction_keepdim_fn(a))


def test_reduction_handles_sum_strided():
    base = torch.randn(3, 4, dtype=torch.float32)
    a = base.t()
    _assert_codegen_source_matches(
        "reduction_sum_strided.c",
        BACKEND.get_generic_source,
        reduction_strided_fn,
        (a,),
    )
    compiled = torch.compile(reduction_strided_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, reduction_strided_fn(a))


def test_reduction_handles_sum_broadcast_producer():
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    b = torch.randn(1, 4, dtype=torch.float32)
    _assert_codegen_source_matches(
        "reduction_sum_broadcast.c",
        BACKEND.get_generic_source,
        reduction_broadcast_fn,
        (a, b),
    )
    compiled = torch.compile(
        reduction_broadcast_fn, backend=BACKEND.codegen_generic_backend
    )
    result = compiled(a, b)
    torch.testing.assert_close(result, reduction_broadcast_fn(a, b))


def test_arg_reduction_handles_argmax_dim():
    a = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "arg_reduction_argmax_dim.c",
        BACKEND.get_generic_source,
        argmax_dim_fn,
        (a,),
    )
    compiled = torch.compile(argmax_dim_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, argmax_dim_fn(a))


def test_elementwise_kernel_source_matches_expected():
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "unary_atan_single.c", BACKEND.get_generic_source, atan_fn, (a,)
    )
    _assert_codegen_source_matches(
        "binary_add_single.c",
        BACKEND.get_generic_source,
        add_broadcast_fn,
        (a, b),
    )


def test_binary_temp_allocations_exceed_threshold():
    backend = CodegenBackend(temp_allocation_threshold=64)
    a = torch.randn(1, dtype=torch.float32)
    b = torch.randn(1, dtype=torch.float32)
    c = torch.randn(1, 2, 2, 5, dtype=torch.float32)
    d = torch.randn(1, 2, 2, 5, dtype=torch.float32)
    _assert_codegen_source_matches(
        "binary_temp_alloc.c",
        backend.get_generic_source,
        temp_alloc_fn,
        (a, b, c, d),
    )


@pytest.mark.parametrize(
    ("reference_name", "fn", "inputs_factory"),
    [
        ("fill_full_like.c", fill_full_like_fn, lambda: (torch.randn(2, 3),)),
        ("view_reshape.c", view_reshape_fn, lambda: (torch.randn(2, 3, 4),)),
        ("flip_flip.c", flip_fn, lambda: (torch.randn(2, 3),)),
        ("repeat_repeat.c", repeat_fn, lambda: (torch.randn(2, 3),)),
        ("arange_arange.c", arange_fn, lambda: (0, 6, 1)),
        ("softmax_softmax.c", softmax_fn, lambda: (torch.randn(2, 3),)),
        ("cumsum_cumsum.c", cumsum_fn, lambda: (torch.randn(2, 3),)),
        ("diagonal_diagonal.c", diagonal_fn, lambda: (torch.randn(3, 3),)),
        (
            "addmm_addmm.c",
            addmm_fn,
            lambda: (torch.randn(2, 2), torch.randn(2, 3), torch.randn(3, 2)),
        ),
        (
            "addbmm_addbmm.c",
            addbmm_fn,
            lambda: (torch.randn(2, 4), torch.randn(5, 2, 3), torch.randn(5, 3, 4)),
        ),
        (
            "addmv_addmv.c",
            addmv_fn,
            lambda: (torch.randn(2), torch.randn(2, 3), torch.randn(3)),
        ),
        (
            "addr_addr.c",
            addr_fn,
            lambda: (torch.randn(2, 3), torch.randn(2), torch.randn(3)),
        ),
        (
            "matmul_matmul.c",
            matmul_fn,
            lambda: (torch.randn(2, 3), torch.randn(3, 4)),
        ),
        (
            "linear_linear.c",
            linear_fn,
            lambda: (torch.randn(2, 3), torch.randn(4, 3), torch.randn(4)),
        ),
        (
            "conv1d_conv1d.c",
            conv1d_fn,
            lambda: (
                torch.randn(1, 2, 5),
                torch.randn(3, 2, 3),
                torch.randn(3),
            ),
        ),
        ("pool1d_max_pool1d.c", max_pool1d_fn, lambda: (torch.randn(1, 2, 4),)),
        (
            "pool3d_max_pool3d.c",
            max_pool3d_fn,
            lambda: (torch.randn(1, 2, 4, 4, 4),),
        ),
        (
            "pool2d_backward_max_pool2d.c",
            max_pool2d_backward_fn,
            lambda: (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 2, 2)),
        ),
        (
            "embedding_embedding.c",
            embedding_fn,
            lambda: (torch.randn(4, 3), torch.tensor([[0, 1], [2, 3]])),
        ),
        (
            "embedding_bag_embedding_bag.c",
            embedding_bag_fn,
            lambda: (
                torch.randn(4, 3),
                torch.tensor([0, 1, 2, 3], dtype=torch.int64),
                torch.tensor([0, 2], dtype=torch.int64),
            ),
        ),
        (
            "embedding_dense_backward_embedding_dense_backward.c",
            embedding_dense_backward_fn,
            lambda: (
                torch.randn(2, 2, 3),
                torch.tensor([[0, 1], [2, 3]]),
            ),
        ),
        (
            "gather_gather.c",
            gather_fn,
            lambda: (torch.randn(2, 3), torch.tensor([[0, 1], [1, 2]])),
        ),
        (
            "index_put_index_put.c",
            index_put_fn,
            lambda: (
                torch.randn(3),
                torch.tensor([0, 2], dtype=torch.int64),
                torch.randn(2),
            ),
        ),
        (
            "index_select_index_select.c",
            index_select_fn,
            lambda: (torch.randn(4, 3), torch.tensor([0, 2], dtype=torch.int64)),
        ),
        (
            "batch_norm_native_batch_norm.c",
            batch_norm_fn,
            lambda: (
                torch.randn(2, 3, 2, 2),
                torch.randn(3),
                torch.randn(3),
                torch.zeros(3),
                torch.ones(3),
            ),
        ),
        (
            "layer_norm_layer_norm.c",
            layer_norm_fn,
            lambda: (torch.randn(2, 4), torch.randn(4), torch.randn(4)),
        ),
        (
            "layer_norm_backward_layer_norm_backward.c",
            layer_norm_backward_fn,
            lambda: (
                torch.randn(2, 4),
                torch.randn(4),
                torch.randn(4),
                torch.randn(2, 1),
                torch.randn(2, 1),
                torch.randn(2, 4),
            ),
        ),
        (
            "group_norm_group_norm.c",
            group_norm_fn,
            lambda: (torch.randn(2, 2, 2, 2), torch.randn(2), torch.randn(2)),
        ),
        (
            "group_norm_backward_group_norm_backward.c",
            group_norm_backward_fn,
            lambda: (
                torch.randn(2, 2, 2, 2),
                torch.randn(2),
                torch.randn(2, 1),
                torch.randn(2, 1),
                torch.randn(2, 2, 2, 2),
            ),
        ),
        ("pdist_pdist.c", pdist_fn, lambda: (torch.randn(4, 3),)),
        (
            "cdist_cdist.c",
            cdist_fn,
            lambda: (torch.randn(2, 3), torch.randn(4, 3)),
        ),
        ("pad_constant_pad_nd.c", pad_fn, lambda: (torch.randn(2, 2),)),
        ("scalar_tensor_scalar_tensor.c", scalar_tensor_fn, lambda: (3.0,)),
        ("random_rand.c", random_fn, lambda: (torch.Size([2, 3]),)),
        ("randperm_randperm.c", randperm_fn, lambda: (5,)),
        ("resize_resize.c", resize_fn, lambda: (torch.randn(6),)),
        (
            "masked_scatter_masked_scatter.c",
            masked_scatter_fn,
            lambda: (
                torch.randn(2, 2),
                torch.tensor([[True, False], [False, True]]),
                torch.randn(2),
            ),
        ),
        (
            "select_scatter_select_scatter.c",
            select_scatter_fn,
            lambda: (torch.randn(2, 3), torch.randn(3)),
        ),
        (
            "scatter_scatter_src.c",
            scatter_fn,
            lambda: (
                torch.randn(2, 3),
                torch.tensor([[0, 1], [1, 2]], dtype=torch.int64),
                torch.randn(2, 2),
            ),
        ),
        (
            "split_with_sizes_split_with_sizes.c",
            split_with_sizes_fn,
            lambda: (torch.randn(2, 3),),
        ),
        ("nonzero_nonzero.c", nonzero_fn, lambda: (torch.randn(2, 2),)),
        ("dropout_dropout.c", dropout_fn, lambda: (torch.randn(2, 2),)),
        ("sort_sort.c", sort_fn, lambda: (torch.randn(2, 3),)),
    ],
)
def test_opkind_smoke_golden_references(reference_name, fn, inputs_factory):
    example_inputs = inputs_factory()
    _assert_codegen_source_matches(
        reference_name, BACKEND.get_generic_source, fn, example_inputs
    )


def test_empty_strided_golden_reference():
    graph = torch.fx.Graph()
    node = graph.call_function(
        torch.ops.aten.empty_strided.default,
        args=([2, 3], [3, 1]),
        kwargs={"dtype": torch.float32},
    )
    graph.output(node)
    gm = torch.fx.GraphModule({}, graph)
    _assert_codegen_graph_matches(
        "empty_strided_empty_strided.c", BACKEND.get_generic_source, gm, ()
    )


def test_col2im_handler_unavailable():
    a = torch.randn(1, 1, 4)
    gm = torch.fx.symbolic_trace(col2im_fn)
    with pytest.raises(CodegenBackendError, match="col2im handler is not available"):
        BACKEND.get_generic_source(gm, (a,))
