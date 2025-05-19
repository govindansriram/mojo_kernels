from GEMM.naive import gpu_unsafe_naive_gemm
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

    print(bench)
