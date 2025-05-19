from GEMM.naive import gpu_unsafe_naive_gemm, gpu_unsafe_naive_gemm2
from gpu.host import DeviceContext
from sys import has_accelerator
from random import rand
from memory import memset_zero
from helpers import fill, host_gemm
from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    ThroughputMeasure,
    BenchMetric,
    Format,
)


alias float_dtype = DType.float32
alias M = 1000
alias N = 1000
alias K = 1000

alias M2 = 4096
alias N2 = 4096
alias K2 = 4096

@parameter
@always_inline
fn benchmark_host_gemm(mut b: Bencher) capturing raises:
    ctx = DeviceContext()

    A_host = ctx.enqueue_create_host_buffer[float_dtype](M * K)
    B_host = ctx.enqueue_create_host_buffer[float_dtype](K * N)
    C_host = ctx.enqueue_create_host_buffer[float_dtype](M * N)
    ctx.synchronize()

    fill(A_host.unsafe_ptr(), M * K)
    fill(B_host.unsafe_ptr(), K * N)
    memset_zero(C_host.unsafe_ptr(), M * N)

    @parameter
    fn run_host_gemm():
        host_gemm[float_dtype](
            A_host.unsafe_ptr(),
            B_host.unsafe_ptr(),
            C_host.unsafe_ptr(),
            M,
            N,
            K,
            K,
            N,
            N,
            1,
            1,
        )

    b.iter[run_host_gemm]()


@parameter
@always_inline
fn benchmark_device_naive_gemm(mut b: Bencher) capturing raises:
    ctx = DeviceContext()

    A_host = ctx.enqueue_create_host_buffer[float_dtype](M * K)
    B_host = ctx.enqueue_create_host_buffer[float_dtype](K * N)
    C_host = ctx.enqueue_create_host_buffer[float_dtype](M * N)
    ctx.synchronize()

    fill(A_host.unsafe_ptr(), M * K)
    fill(B_host.unsafe_ptr(), K * N)
    memset_zero(C_host.unsafe_ptr(), M * N)

    A_device = ctx.enqueue_create_buffer[float_dtype](M * K)
    B_device = ctx.enqueue_create_buffer[float_dtype](K * N)
    C_device = ctx.enqueue_create_buffer[float_dtype](M * N)

    ctx.enqueue_copy(A_device, A_host)
    ctx.enqueue_copy(B_device, B_host)
    ctx.enqueue_copy(C_device, C_host)

    @parameter
    fn run_naive_gemm(ctx: DeviceContext) raises:
        gpu_unsafe_naive_gemm[float_dtype](
            ctx,
            A_device,
            B_device,
            C_device,
            M,
            N,
            K,
            K,
            N,
            N,
            1,
            1,
        )

    b.iter_custom[run_naive_gemm](ctx)


@parameter
@always_inline
fn benchmark_device_large_naive_gemm(mut b: Bencher) capturing raises:
    ctx = DeviceContext()

    A_host = ctx.enqueue_create_host_buffer[float_dtype](M2 * K2)
    B_host = ctx.enqueue_create_host_buffer[float_dtype](K2 * N2)
    C_host = ctx.enqueue_create_host_buffer[float_dtype](M2 * N2)
    ctx.synchronize()

    fill(A_host.unsafe_ptr(), M2 * K2)
    fill(B_host.unsafe_ptr(), K2 * N2)
    memset_zero(C_host.unsafe_ptr(), M2 * N2)

    A_device = ctx.enqueue_create_buffer[float_dtype](M2 * K2)
    B_device = ctx.enqueue_create_buffer[float_dtype](K2 * N2)
    C_device = ctx.enqueue_create_buffer[float_dtype](M2 * N2)

    ctx.enqueue_copy(A_device, A_host)
    ctx.enqueue_copy(B_device, B_host)
    ctx.enqueue_copy(C_device, C_host)

    @parameter
    fn run_naive_gemm(ctx: DeviceContext) raises:
        gpu_unsafe_naive_gemm[float_dtype](
            ctx,
            A_device,
            B_device,
            C_device,
            M2,
            N2,
            K2,
            K2,
            N2,
            N2,
            1,
            1,
        )

    b.iter_custom[run_naive_gemm](ctx)


@parameter
@always_inline
fn benchmark_device_large_naive_gemm2(mut b: Bencher) capturing raises:
    ctx = DeviceContext()

    A_host = ctx.enqueue_create_host_buffer[float_dtype](M2 * K2)
    B_host = ctx.enqueue_create_host_buffer[float_dtype](K2 * N2)
    C_host = ctx.enqueue_create_host_buffer[float_dtype](M2 * N2)
    ctx.synchronize()

    fill(A_host.unsafe_ptr(), M2 * K2)
    fill(B_host.unsafe_ptr(), K2 * N2)
    memset_zero(C_host.unsafe_ptr(), M2 * N2)

    A_device = ctx.enqueue_create_buffer[float_dtype](M2 * K2)
    B_device = ctx.enqueue_create_buffer[float_dtype](K2 * N2)
    C_device = ctx.enqueue_create_buffer[float_dtype](M2 * N2)

    ctx.enqueue_copy(A_device, A_host)
    ctx.enqueue_copy(B_device, B_host)
    ctx.enqueue_copy(C_device, C_host)

    @parameter
    fn run_naive_gemm(ctx: DeviceContext) raises:
        gpu_unsafe_naive_gemm2[float_dtype](
            ctx,
            A_device,
            B_device,
            C_device,
            M2,
            N2,
            K2,
            K2,
            N2,
            N2,
            1,
            1,
        )

    b.iter_custom[run_naive_gemm](ctx)


def main():
    # needed to visualize print statements
    var bench = Bench(BenchConfig(max_iters=10))

    bench.bench_function[benchmark_host_gemm](
        BenchId("gemm_naive", "cpu"),
        # ThroughputMeasure(BenchMetric.elements, 10),
        ThroughputMeasure(BenchMetric.flops, 2 * M * N * K),
    )

    bench.bench_function[benchmark_device_naive_gemm](
        BenchId("gemm_naive", "gpu"),
        # ThroughputMeasure(BenchMetric.elements, 10),
        ThroughputMeasure(BenchMetric.flops, 2 * M * N * K),
    )

    bench.bench_function[benchmark_device_large_naive_gemm](
        BenchId("gemm_naive_large1", "gpu"),
        # ThroughputMeasure(BenchMetric.elements, 10),
        ThroughputMeasure(BenchMetric.flops, 2 * M2 * N2 * K2),
    )

    bench.bench_function[benchmark_device_large_naive_gemm2](
        BenchId("gemm_naive_large2", "gpu"),
        # ThroughputMeasure(BenchMetric.elements, 10),
        ThroughputMeasure(BenchMetric.flops, 2 * M2 * N2 * K2),
    )

    print(bench)
