################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
Overlapping AllGather GEMM
==========================

In this tutorial, you will write a simple Allgather GEMM fusion kernel using Triton-distributed.

In doing so, you will learn about:

* Writing a GEMM kernel that consume the results of AllGather.

* Optimizing the internode communication with 2D Allgather.

    # To run this tutorial
    source ./scripts/sentenv.sh
    bash ./scripts/launch.sh ./tutorials/07-overlapping-allgather-gemm.py

"""
import argparse
import os
import time
from functools import partial
from typing import Optional

import nvshmem.core
import torch
import torch.distributed
from cuda import cudart

import flux
from flux.testing import (
    DTYPE_MAP,
    RING_MODE_MAP,
    all_gather_into_tensor_with_fp8,
    generate_data,
    initialize_distributed,
    zeros_with_fp8,
    matmul_int8,
)
import flux.testing
from flux.testing.perf_db_helper import should_log_to_rds, set_global_args, log_perf
from flux.util import bench_func, is_fp8_dtype

try:
    from flux.triton.ag_gemm import AgGemmTriton
except Exception as e:
    print("triton module import failed. skip...")

import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.kernels.common_ops import set_signal, wait_eq
from triton_dist.kernels.nvidia.allgather_gemm import create_ag_gemm_context
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import initialize_distributed, nvshmem_barrier_all_on_stream

print = partial(print, flush=True)


class PerfResult:
    def __init__(
        self,
        name: str,
        output: torch.Tensor,
        gathered_output: torch.Tensor,
        total_ms: float,
        time1: str,
        gemm_time_ms: float,
        time2: str,
        comm_time_ms: float,
        time3: str = "gemm_only",
        gemm_only_time_ms: float = 0,
    ) -> None:
        self.name = name
        self.output = output
        self.gathered_output = gathered_output
        self.total_ms = total_ms
        self.time1 = time1
        self.time2 = time2
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms
        self.time3 = time3
        self.gemm_only_time_ms = gemm_only_time_ms

    def __repr__(self) -> str:
        if self.gemm_only_time_ms == 0.0:
            if self.gemm_time_ms == 0.0:
                    return (
                    f"{self.name}: total {self.total_ms:.3f} ms"
                )
            else:
                return (
                    f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                    f", {self.time2} {self.comm_time_ms:.3f} ms"
                )
        else:
            return (
                f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                f", {self.time2} {self.comm_time_ms:.3f} ms, {self.time3} {self.gemm_only_time_ms:.3f} ms"
            )

@torch.no_grad()
def perf_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    is_fp8: bool,
    is_s8_dequant: bool,
    warmup: int,
    iters: int,
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()

    torch.distributed.barrier()
    # All gather input tensors from all gpus
    full_input = zeros_with_fp8(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    full_input_scale = (
        torch.zeros(
            (M, 1), dtype=input_scale.dtype, device=torch.cuda.current_device(), requires_grad=False
        )
        if is_s8_dequant
        else None
    )

    alpha_scale = 1.0
    if is_fp8:
        alpha_scale = input_scale * weight_scale
        input = input.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        full_input = full_input.to(torch.bfloat16)

    if is_s8_dequant:
        assert input_scale is not None
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)
    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)

    torch.distributed.barrier()
    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    allgather_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
        allgather_end_events[i].record()
        if is_s8_dequant:
            accum = matmul_int8(full_input, weight.t()).to(torch.float32)
            output = full_input_scale * weight_scale * accum
        else:
            output = alpha_scale * torch.matmul(full_input, weight.t())

        if is_fp8 or is_s8_dequant:
            output = output.to(torch.bfloat16)
        if bias is not None:
            output += bias
        end_events[i].record()

    comm_times = []  # all gather
    gemm_times = []  # gemm
    for i in range(total_iters):
        allgather_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(allgather_end_events[i]) / 1000)
            gemm_times.append(allgather_end_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000
    gemm_time = sum(gemm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        output=output,
        gathered_output=full_input,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )

@torch.no_grad()
def perf_triton_dist(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    is_fp8: bool,
    is_s8_dequant: bool,
    warmup: int,
    iters: int,

    rank: int,
    num_ranks: int,
    workspace_tensors: list[torch.Tensor],
    barrier_tensors: list[torch.Tensor],
    comm_buf: torch.Tensor,
    ag_stream: Optional[torch.cuda.Stream] = None,
    internode_ag_stream: Optional[torch.cuda.Stream] = None,
    local_world_size: int = 8,
    signal_target: int = 1,
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()

    torch.distributed.barrier()
    # All gather input tensors from all gpus
    full_input = zeros_with_fp8(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    # Launch AllGather communication with error handling
    inter_node_allgather(
        input, full_input, barrier_tensors, signal_target,
        rank, local_world_size, num_ranks, ag_stream,
        internode_ag_stream
    )

    torch.distributed.barrier()
    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    allgather_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        # Launch AllGather communication with error handling
        inter_node_allgather(
            input, full_input, barrier_tensors, signal_target,
            rank, local_world_size, num_ranks, ag_stream,
            internode_ag_stream
        )
        allgather_end_events[i].record()
        output = triton_matmul(full_input, weight.t())
        end_events[i].record()

    
    comm_times = []  # Comm
    gemm_times = []  # gemm

    for i in range(total_iters):
        allgather_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(allgather_end_events[i]) / 1000)
            gemm_times.append(allgather_end_events[i].elapsed_time(end_events[i]) / 1000)    

    comm_time = sum(comm_times) / iters * 1000
    gemm_time = sum(gemm_times) / iters * 1000

    return PerfResult(
        name=f"triton_dist #{TP_GROUP.rank()}",
        output=output,
        gathered_output=full_input,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_flux(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    transpose_weight: bool = True,
    gather_input: bool = False,
    ring_mode: Optional[flux.AGRingMode] = None,
    warmup: int = 5,
    iters: int = 10,
    fast_acc: bool = False,
    verify: bool = False,
    use_cuda_core_local: bool = False,
    use_cuda_core_ag: bool = False,
    use_pdl: bool = False,
):
    input_dtype = input.dtype
    is_fp8 = is_fp8_dtype(input_dtype)
    is_s8_dequant = input_dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else input_dtype
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()
    K = input.size(1)

    if transpose_weight:
        w = weight.t().contiguous()
        N = w.size(1)
    else:
        w = weight
        N = w.size(0)

    torch.distributed.barrier()
    full_input = zeros_with_fp8(
        (M, K),
        dtype=input_dtype,
        device=torch.cuda.current_device(),
    )

    full_input_scale = (
        torch.zeros((M, 1), dtype=input_scale.dtype, device=torch.cuda.current_device())
        if is_s8_dequant
        else None
    )
    all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
    if is_s8_dequant:
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)

    use_fp8_gemm = True if is_fp8 else False
    gemm_only_op = flux.GemmOnly(
        input_dtype=input_dtype,
        weight_dtype=input_dtype,
        output_dtype=output_dtype,
        transpose_weight=transpose_weight,
        use_fp8_gemm=use_fp8_gemm,
    )
    gemm_only_output = torch.empty(
        [M, N], dtype=output_dtype, device=input.device, requires_grad=False
    )

    ag_gemm_output = torch.empty([M, N], dtype=output_dtype, device=input.device)
    ag_option = flux.AllGatherOption()
    ag_option.mode = ring_mode
    all_gather_gemm_kernel = flux.AGKernel(
        TP_GROUP,
        NNODES,
        M,
        N,
        K,
        input_dtype,
        output_dtype=output_dtype,
        use_pdl=use_pdl,
    )

    warmup_iters = warmup
    total_iters = warmup_iters + iters if not verify else 1
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        gemm_only_output = gemm_only_op.forward(
            full_input,
            w,
            bias=bias,
            output_buf=gemm_only_output,
            input_scale=input_scale if not is_s8_dequant else full_input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
        )
        end_events[i].record()
    torch.cuda.current_stream().synchronize()

    gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)
    gemm_time = sum(gemm_times)

    full_input.zero_()
    time.sleep(1)

    torch.distributed.barrier()
    ag_option.use_cuda_core_local = use_cuda_core_local
    ag_option.use_cuda_core_ag = use_cuda_core_ag
    for i in range(total_iters):
        start_events[i].record()
        all_gather_gemm_kernel.forward(
            input,
            w,
            bias=bias,
            output=ag_gemm_output,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
            gathered_input=full_input if gather_input else None,
            transpose_weight=transpose_weight,
            all_gather_option=ag_option,
        )
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    ag_gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            ag_gemm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    ag_gemm_time = sum(ag_gemm_times)

    ## signals are already set
    for i in range(total_iters):
        start_events[i].record()
        if not verify:
            _ = all_gather_gemm_kernel.gemm_only(
                full_input,
                w,
                bias=bias,
                input_scale=full_input_scale,
                weight_scale=weight_scale,
                output_scale=None,
                fast_accum=fast_acc,
                transpose_weight=transpose_weight,
            )
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    gemm_only_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_only_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)
    gemm_only_time = sum(gemm_only_times)

    ag_gemm_time_ms = ag_gemm_time / iters * 1000
    gemm_time_ms = gemm_time / iters * 1000
    comm_time_ms = (ag_gemm_time - gemm_time) / iters * 1000
    gemm_only_time_ms = gemm_only_time / iters * 1000

    return PerfResult(
        name=f"flux  #{TP_GROUP.rank()}",
        output=ag_gemm_output,
        gathered_output=full_input,
        total_ms=ag_gemm_time_ms,
        time1="gemm",
        gemm_time_ms=gemm_time_ms,
        time2="comm",
        comm_time_ms=comm_time_ms,
        time3="gemm_only",
        gemm_only_time_ms=gemm_only_time_ms,
    )

@torch.no_grad()
def perf_flux_no_overlap(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    transpose_weight: bool = True,
    gather_input: bool = True,  # not used. always as true
    warmup: int = 5,
    iters: int = 10,
    fast_acc: bool = False,
):
    input_dtype = input.dtype
    is_fp8 = is_fp8_dtype(input_dtype)
    is_s8_dequant = input_dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else input.dtype
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()
    K = input.size(1)

    if transpose_weight:
        w = weight.t().contiguous()
        N = w.size(1)
    else:
        w = weight
        N = w.size(0)

    full_input = zeros_with_fp8(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    full_input_scale = (
        torch.zeros((M, 1), dtype=input_scale.dtype, device=torch.cuda.current_device())
        if is_s8_dequant
        else None
    )
    all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
    if is_s8_dequant:
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)

    ag_gemm_op = flux.AGKernel(
        TP_GROUP,
        NNODES,
        M,
        N,
        K,
        input_dtype,
        output_dtype=output_dtype,
    )

    gemm_only_output = torch.empty(
        [M, N], dtype=output_dtype, device=input.device, requires_grad=False
    )

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    allgather_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
        allgather_end_events[i].record()

        gemm_only_output = ag_gemm_op.gemm_only(
            full_input,
            w,
            bias=bias,
            input_scale=full_input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
            transpose_weight=transpose_weight,
        )
        end_events[i].record()

    comm_times = []  # all gather
    gemm_times = []  # gemm
    for i in range(total_iters):
        allgather_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(allgather_end_events[i]) / 1000)
            gemm_times.append(allgather_end_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000
    gemm_time = sum(gemm_times) / iters * 1000

    return PerfResult(
        name=f"flux(no-overlap) #{TP_GROUP.rank()}",
        output=gemm_only_output,
        gathered_output=full_input,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


# %%
# Now, let's write a GEMM kernel to consume the transfered tensors!
# We use tma to optimize the GEMM performance on the SM90 platform


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def kernel_consumer_gemm_persistent(
        a_ptr,
        b_ptr,
        c_ptr,  #
        M,
        N,
        K,  #
        rank: tl.constexpr,
        num_ranks: tl.constexpr,
        ready_ptr,
        comm_buf_ptr,
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        EPILOGUE_SUBTILE: tl.constexpr,  #
        NUM_SMS: tl.constexpr,
        ready_value: tl.constexpr = 1,
        local_world_size: tl.constexpr = 8):  #
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    node_id = rank // local_world_size
    nnodes = num_ranks // local_world_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // num_ranks
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # swizzle m
            if nnodes == 1:
                alpha = 0
                beta = 0
                pid_m = (pid_m + ((((rank ^ alpha) + beta) % num_ranks) *
                                  pid_ms_per_rank)) % num_pid_m
            else:
                m_rank = pid_m // pid_ms_per_rank
                pid_m_intra_rank = pid_m - m_rank * pid_ms_per_rank
                m_node_id = m_rank // local_world_size
                m_local_rank = m_rank % local_world_size
                swizzle_m_node_id = (m_node_id + node_id) % nnodes
                swizzle_m_local_rank = (m_local_rank + rank) % local_world_size
                swizzle_m_rank = swizzle_m_node_id * local_world_size + swizzle_m_local_rank

                pid_m = swizzle_m_rank * pid_ms_per_rank + pid_m_intra_rank

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            rank_beg = offs_am // M_per_rank
            rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // M_per_rank
            # Each tile wait for the corresponding data to ready
            token = dl.wait(ready_ptr + rank_beg,
                            rank_end - rank_beg + 1,
                            "gpu",
                            "acquire",
                            waitValue=ready_value)
            a_desc = dl.consume_token(a_desc, token)

        offs_k = ki * BLOCK_SIZE_K
        # Iteration along k-dimension, and performing multiply.
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator,
                                 (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                   dtype=tl.float32)


# %%
# To fully utilize the bandwidth, internode AllGather Kernel is composed of two parts considering the bandwidth gap between intra-node links and inter-node links.
# --------------


def inter_node_allgather(local_tensor: torch.Tensor,
                         ag_buffer: list[torch.Tensor],
                         signal_buffer: list[torch.Tensor], signal_target,
                         rank, local_world_size, world_size,
                         intranode_ag_stream, internode_ag_stream):
    local_rank = rank % local_world_size
    n_nodes = world_size // local_world_size
    M_per_rank, N = local_tensor.shape

    # Each rank sends the local_tensor to ranks of other nodes with the same local_rank
    # Assuming there are 2 nodes, each with 4 workers
    # 0-th local tensor ([0] -> [4]), 4-th local tensor ([4] -> [0])
    # 1-th local tensor ([1] -> [5]), 5-th local tensor ([5] -> [1])
    # 2-th local tensor ([2] -> [6]), 6-th local tensor ([6] -> [2])
    # 3-th local tensor ([3] -> [7]), 7-th local tensor ([7] -> [3])
    with torch.cuda.stream(internode_ag_stream):
        grid = lambda META: (int(n_nodes - 1), )
        nvshmem_device_producer_p2p_put_block_kernel[grid](
            ag_buffer[local_rank],
            signal_buffer[local_rank],
            M_per_rank * N,
            local_tensor.element_size(),
            signal_target,
            rank,
            local_world_size,
            world_size,
            num_warps=32,  # each sm launches 1024 threads
        )

    # Each rank sends the local_tensor and the received internode tensors to intranode ranks.
    # 0-th and 4-th local tensors ([0]->[1,2,3])
    # 1-th and 5-th local tensors ([1]->[0,2,3])
    # 2-th and 6-th local tensors ([2]->[0,1,3])
    # 3-th and 7-th local tensors ([3]->[0,1,2])
    # 0-th and 4-th local tensors ([4]->[5,6,7])
    # 1-th and 5-th local tensors ([5]->[4,6,7])
    # 2-th and 6-th local tensors ([6]->[4,5,7])
    # 3-th and 7-th local tensors ([7]->[4,5,6])
    with torch.cuda.stream(intranode_ag_stream):
        cp_engine_producer_all_gather_put(local_tensor, ag_buffer,
                                          signal_buffer, M_per_rank, N,
                                          signal_target, rank,
                                          local_world_size, world_size,
                                          intranode_ag_stream)

    intranode_ag_stream.wait_stream(internode_ag_stream)


# %%
# Let's declare a function to perform internode communication.


@triton.jit
def nvshmem_device_producer_p2p_put_block_kernel(
    ag_buffer_ptr,  # *Pointer* to allgather output vector. The rank-th index has been loaded with local tensor
    signal_buffer_ptr,  # *Pointer* to signal barrier.
    elem_per_rank,
    size_per_elem,
    signal_target,
    rank,
    local_world_size,
    world_size,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    n_nodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_rank = rank // local_world_size

    for i in range(pid, n_nodes - 1, num_pid):
        # Each SM is assigned to one peer.
        # Peer id is caculated based on pid and local_rank.
        peer = local_rank + (node_rank + i + 1) % n_nodes * local_world_size
        # We use putmem_signal_block to send data and notify the peer.
        # Since this is the allgather operation, the offsets of both src and dst tensor are both *rank*.
        libshmem_device.putmem_signal_block(
            ag_buffer_ptr + rank * elem_per_rank,
            ag_buffer_ptr + rank * elem_per_rank,
            elem_per_rank * size_per_elem,
            signal_buffer_ptr + rank,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


# %%
# Let's also declare a function to perform intranode communication.


def cp_engine_producer_all_gather_put(local_tensor, ag_buffer, signal_buffer,
                                      M_per_rank, N, signal_target, rank,
                                      local_world_size, world_size,
                                      intranode_ag_stream):
    local_rank = rank % local_world_size
    n_nodes = world_size // local_world_size
    node_rank = rank // local_world_size

    for i in range(1, local_world_size):
        segment = rank * M_per_rank * N
        local_dst_rank = (local_rank + local_world_size - i) % local_world_size
        src_ptr = ag_buffer[local_rank].data_ptr(
        ) + segment * local_tensor.element_size()
        dst_ptr = ag_buffer[local_dst_rank].data_ptr(
        ) + segment * local_tensor.element_size()
        # Using copy engine to perform intranode transmission
        # Sending rank-th local tensor to other ranks inside the node.
        (err, ) = cudart.cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            M_per_rank * N * local_tensor.element_size(),
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            intranode_ag_stream.cuda_stream,
        )
        # Notify the peer that the transmission is done.
        # set_signal(signal_buffer[local_dst_rank][rank].data_ptr(),
        #            signal_target, intranode_ag_stream, True)
        set_signal(signal_buffer[local_dst_rank][rank],
                   signal_target, intranode_ag_stream)

    for i in range(1, n_nodes):
        recv_rank = local_rank + (node_rank + n_nodes -
                                  i) % n_nodes * local_world_size
        recv_segment = recv_rank * M_per_rank * N
        # Waiting for the internode data ready
        wait_eq(signal_buffer[local_rank][recv_rank].data_ptr(), signal_target,
                intranode_ag_stream, True)
        src_ptr = ag_buffer[local_rank].data_ptr(
        ) + recv_segment * local_tensor.element_size()
        for j in range(1, local_world_size):
            local_dst_rank = (local_rank + local_world_size -
                              j) % local_world_size
            dst_ptr = ag_buffer[local_dst_rank].data_ptr(
            ) + recv_segment * local_tensor.element_size()
            # Sending (local_rank + j*local_world_size) % world_size -th local tensor to other ranks inside the node.
            (err, ) = cudart.cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                M_per_rank * N * local_tensor.element_size(),
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                intranode_ag_stream.cuda_stream,
            )
            # Notify the peer that the transmission is done.
            set_signal(signal_buffer[local_dst_rank][recv_rank].data_ptr(),
                       signal_target, intranode_ag_stream, True)


# %%
# Kernel Integration and Orchestration
# ====================================
#
# This section combines all previously defined kernels into a unified operation:
# - AllGather communication kernels (inter-node and intra-node)
# - GEMM computation kernel with persistent execution
# - Synchronization and memory management
#
# The ag_gemm_persistent_op function orchestrates the overlapping of communication
# and computation for optimal performance in distributed matrix multiplication.


def ag_gemm_persistent_op(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    rank: int,
    num_ranks: int,
    workspace_tensors: list[torch.Tensor],
    barrier_tensors: list[torch.Tensor],
    comm_buf: torch.Tensor,
    ag_stream: Optional[torch.cuda.Stream] = None,
    internode_ag_stream: Optional[torch.cuda.Stream] = None,
    BLOCK_M: int = 128,
    BLOCK_N: int = 256,
    BLOCK_K: int = 64,
    stages: int = 3,
    local_world_size: int = 8,
    signal_target: int = 1,
) -> triton.compiler.CompiledKernel:
    """
    Orchestrates overlapping AllGather communication and GEMM computation.
    
    This function combines inter-node and intra-node AllGather operations with
    a persistent GEMM kernel to achieve optimal performance in distributed
    matrix multiplication scenarios.
    
    Args:
        a: Local input tensor of shape (M_per_rank, K)
        b: Weight tensor of shape (N_per_rank, K)
        c: Output tensor of shape (M, N_per_rank)
        rank: Current process rank
        num_ranks: Total number of processes
        workspace_tensors: List of workspace tensors for AllGather
        barrier_tensors: List of synchronization barrier tensors
        comm_buf: Communication buffer tensor
        ag_stream: AllGather stream (created if None)
        internode_ag_stream: Inter-node AllGather stream
        BLOCK_M: Block size for M dimension
        BLOCK_N: Block size for N dimension
        BLOCK_K: Block size for K dimension
        stages: Number of pipeline stages
        local_world_size: Number of processes per node
        signal_target: Signal value for synchronization
        
    Returns:
        Compiled Triton kernel object
        
    Raises:
        ValueError: If tensor shapes, dtypes, or configuration parameters are invalid
        RuntimeError: If AllGather operation or kernel launch fails
    """
    # Input validation
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Incompatible K dimensions: a.shape[1]={a.shape[1]}, b.shape[1]={b.shape[1]}")
    
    if a.dtype != b.dtype:
        raise ValueError(f"Incompatible dtypes: a.dtype={a.dtype}, b.dtype={b.dtype}")
    
    if num_ranks % local_world_size != 0:
        raise ValueError(f"num_ranks ({num_ranks}) must be divisible by local_world_size ({local_world_size})")
    
    if rank >= num_ranks or rank < 0:
        raise ValueError(f"rank ({rank}) must be in range [0, {num_ranks})")

    # Extract tensor dimensions
    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    N_per_rank, _ = b.shape

    # Calculate distributed topology parameters
    local_rank = rank % local_world_size
    n_nodes = num_ranks // local_world_size
    
    # Resource allocation for communication and computation
    # Use at least 1 SM for AllGather, but ensure we have SMs for GEMM
    num_ag_sms = max(1, n_nodes - 1) if n_nodes > 1 else 0
    device_props = torch.cuda.get_device_properties("cuda")
    total_sms = device_props.multi_processor_count
    
    if num_ag_sms >= total_sms:
        raise ValueError(f"Not enough SMs: need {num_ag_sms} for AllGather, but only {total_sms} available")
    
    num_gemm_sms = total_sms - num_ag_sms

    # Stream management with proper error handling
    if ag_stream is None:
        ag_stream = torch.cuda.Stream()
    
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)

    # Launch AllGather communication with error handling
    try:
        inter_node_allgather(
            a, workspace_tensors, barrier_tensors, signal_target,
            rank, local_world_size, num_ranks, ag_stream,
            internode_ag_stream
        )
    except Exception as e:
        raise RuntimeError(f"AllGather operation failed: {e}") from e

    # Configure Triton memory allocator
    def triton_alloc_fn(size: int, alignment: int, stream: Optional[int]) -> torch.Tensor:
        """Custom allocator for Triton kernels."""
        try:
            return torch.empty(size, device="cuda", dtype=torch.int8)
        except Exception as e:
            raise RuntimeError(f"Failed to allocate {size} bytes for Triton kernel: {e}") from e

    triton.set_allocator(triton_alloc_fn)
    
    # Calculate grid dimensions for GEMM kernel
    def calculate_grid(META: dict) -> tuple[int]:
        """Calculate grid dimensions based on tensor sizes and block sizes."""
        num_blocks_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
        num_blocks_n = triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"])
        total_blocks = num_blocks_m * num_blocks_n
        return (min(num_gemm_sms, total_blocks),)

    # Launch persistent GEMM kernel with comprehensive error handling
    try:
        compiled_kernel = kernel_consumer_gemm_persistent[calculate_grid](
            workspace_tensors[local_rank][:M],  # Input tensor slice
            b,                                  # Weight tensor
            c,                                  # Output tensor
            M,                                  # Global M dimension
            N_per_rank,                         # Local N dimension
            K,                                  # K dimension
            rank,                               # Current rank
            num_ranks,                          # Total ranks
            barrier_tensors[local_rank],        # Synchronization barriers
            comm_buf,                           # Communication buffer
            BLOCK_M,                            # M block size
            BLOCK_N,                            # N block size
            BLOCK_K,                            # K block size
            8,                                  # Group size M (hardcoded for now)
            False,                              # Epilogue subtile flag
            NUM_SMS=num_gemm_sms,              # Number of SMs for GEMM
            ready_value=signal_target,          # Signal value for synchronization
            num_stages=stages,                  # Pipeline stages
            num_warps=8,                        # Number of warps per block
        )
    except Exception as e:
        raise RuntimeError(f"Failed to launch GEMM kernel: {e}") from e

    # Synchronize streams to ensure proper ordering
    if internode_ag_stream is not None:
        current_stream.wait_stream(internode_ag_stream)
    current_stream.wait_stream(ag_stream)

    return compiled_kernel

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def triton_matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

# Non-overlap baseline implemented with torch
def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    local_input: torch.Tensor,
    local_weight: torch.Tensor,
    ag_out: torch.Tensor,
):
    torch.distributed.all_gather_into_tensor(ag_out, local_input, pg)
    ag_gemm_output = torch.matmul(ag_out, local_weight)
    return ag_gemm_output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=True,
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument(
        "--fastacc",
        default=False,
        action="store_true",
        help="whether to use fast accumulation (FP8 Gemm only)",
    )
    parser.add_argument(
        "--ring_mode",
        default="auto",
        choices=["auto", "all2all", "ring1d", "ring2d"],
        help="ring mode. auto for auto detect",
    )
    parser.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run once to verify correctness",
    )

    parser.add_argument(
        "--use_cuda_core_local",
        action=argparse.BooleanOptionalAction,
        help="use cuda core to impl local copy, auto select if not specified",
    )

    parser.add_argument(
        "--use_cuda_core_ag",
        action=argparse.BooleanOptionalAction,
        help="use cuda core to impl all gather, auto select if not specified",
    )

    parser.add_argument(
        "--use_pdl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use Programmatic Dependent Launch",
    )

    parser.add_argument(
        "--triton",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run with triton kernels",
    )
    parser.add_argument(
        "--gather_input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="gather input",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
    return parser.parse_args()

if __name__ == "__main__":
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "-1"))
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "-1"))

    # if WORLD_SIZE == LOCAL_WORLD_SIZE:
    #     print(
    #         "Skip the test because this should be performed with 2 nodes or higher"
    #     )
    #     import sys
    #     sys.exit()

    # if torch.cuda.get_device_capability()[0] < 9:
    #     print("device_capability:",torch.cuda.get_device_capability()[0])
    #     print("Skip the test because the device is not sm90 or higher")
    #     import sys
    #     sys.exit()

    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()

    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    args = parse_args()
    input_dtype = DTYPE_MAP[args.dtype]

    assert args.M % TP_GROUP.size() == 0
    assert args.N % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_M = args.M // TP_GROUP.size()
    local_N = args.N // TP_GROUP.size()

    M = args.M 
    N = args.N
    K = args.K
    config = {"BM": 128, "BN": 256, "BK": 64, "stage": 3}
    dtype = torch.float16

    assert M % WORLD_SIZE == 0
    assert N % WORLD_SIZE == 0
    M_per_rank = M // WORLD_SIZE
    N_per_rank = N // WORLD_SIZE

    A = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    B = torch.randn([N_per_rank, K], dtype=dtype, device="cuda")

    ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")
    golden = torch_ag_gemm(TP_GROUP, A, B.T, ag_buffer)

    # We can use a context to wrap all the tensors used at runtime.
    # We rely on NVSHMEM to allocate the symmetric memory for communication
    # In practice, the following parts are encapsulated in ag_gemm_inter_node() of triton_dist.kernels.nvidia.allgather_gemm.py

    C = torch.empty([M, N_per_rank], dtype=dtype, device="cuda")
    ctx = create_ag_gemm_context(A,
                                 B,
                                 rank,
                                 WORLD_SIZE,
                                 max_M=M,
                                 BLOCK_M=config["BM"],
                                 BLOCK_N=config["BN"],
                                 BLOCK_K=config["BK"],
                                 stages=config["stage"])
    ctx.symm_barrier.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    # copy local data to the ctx
    ctx.symm_workspace[rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(A)
    # set_signal(ctx.symm_barrier[rank].data_ptr(), 1,
    #            torch.cuda.current_stream(), True)
    set_signal(ctx.symm_barrier[rank], 1,
               torch.cuda.current_stream())

    # launch the ag_gemm kernel
    ag_gemm_persistent_op(A,
                          B,
                          C,
                          ctx.rank,
                          ctx.num_ranks,
                          ctx.symm_workspaces,
                          ctx.symm_barriers,
                          ctx.symm_comm_buf,
                          ag_stream=ctx.ag_intranode_stream,
                          internode_ag_stream=ctx.ag_internode_stream,
                          local_world_size=LOCAL_WORLD_SIZE,
                          signal_target=1)
    
    if torch.allclose(golden, C, atol=1e-3, rtol=1e-3):
        print(f"✅ triton-dist #{TP_GROUP.rank()}: Triton-dist and Torch match")
    else:
        print(f"❌ triton-dist #{TP_GROUP.rank()}: Triton-dist and Torch differ")


    #
    assert args.M % TP_GROUP.size() == 0
    assert args.N % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_M = args.M // TP_GROUP.size()
    local_N = args.N // TP_GROUP.size()

    scale = TP_GROUP.rank() + 1

    data_config = [
        ((local_M, args.K), input_dtype, (0.01 * scale, 0)),  # A
        ((local_N, args.K), input_dtype, (0.01 * scale, 0)),  # B
        (  # bias
            None if not args.has_bias else ((args.M, local_N), input_dtype, (0.1 * scale, 0))
        ),
        None,  # input_scale
        None,  # weight_scale
    ]

    generator = generate_data(data_config)
    input, weight, bias, input_scale, weight_scale = next(generator)

    TP_GROUP.barrier()

    with flux.util.group_profile(
        name="ag_gemm_" + os.environ["TORCHELASTIC_RUN_ID"], do_prof=args.profile, group=TP_GROUP
    ):
        perf_res_torch = perf_torch(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            False,
            False,
            args.warmup,
            args.iters,
        )

        perf_res_triton_dist = perf_triton_dist(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            False,
            False,
            args.warmup,
            args.iters,

            ctx.rank,
            ctx.num_ranks,
            ctx.symm_workspaces,
            ctx.symm_barriers,
            ctx.symm_comm_buf,
            ag_stream=ctx.ag_intranode_stream,
            internode_ag_stream=ctx.ag_internode_stream,
            local_world_size=LOCAL_WORLD_SIZE,
            signal_target=1
        )

        perf_res_flux = perf_flux(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            args.transpose_weight,
            args.gather_input,
            RING_MODE_MAP[args.ring_mode],
            args.warmup,
            args.iters,
            args.fastacc,
            args.verify,
            args.use_cuda_core_local,
            args.use_cuda_core_ag,
            args.use_pdl,
        )

        perf_res_flux_no_overlap = perf_flux_no_overlap(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            args.transpose_weight,
            args.gather_input,  # not used,
            args.warmup,
            args.iters,
            args.fastacc,
        )

    if should_log_to_rds():
        set_global_args("ag_gemm", args)
    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            log_perf(perf_res_torch)
            log_perf(perf_res_triton_dist)
            log_perf(perf_res_flux)
            log_perf(perf_res_flux_no_overlap)
        torch.distributed.barrier()

    torch.distributed.barrier()

    TP_GROUP.barrier()
    torch.cuda.synchronize()

    ctx.finailize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
