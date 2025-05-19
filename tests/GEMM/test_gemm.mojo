from GEMM.naive import gpu_unsafe_naive_gemm
from gpu.host import DeviceContext
from sys import has_accelerator
from random import rand
from helpers import fill_whole, host_gemm, assert_ptr_equality
from memory import memset_zero


alias float_dtype = DType.float32
alias M = 1111
alias N = 1312
alias K = 755


def test_unsafe_naive_gemm():
    
    @parameter
    if not has_accelerator():
        raise Error("No device found")

    ctx = DeviceContext()

    A_host = ctx.enqueue_create_host_buffer[float_dtype](M * K)
    B_host = ctx.enqueue_create_host_buffer[float_dtype](K * N)
    C_host = ctx.enqueue_create_host_buffer[float_dtype](M * N)
    C_host2 = ctx.enqueue_create_host_buffer[float_dtype](M * N)
    ctx.synchronize()

    fill_whole(A_host.unsafe_ptr(), M * K)
    fill_whole(B_host.unsafe_ptr(), K * N)
    memset_zero(C_host.unsafe_ptr(), M * N)
    memset_zero(C_host2.unsafe_ptr(), M * N)

    A_device = ctx.enqueue_create_buffer[float_dtype](M * K)
    B_device = ctx.enqueue_create_buffer[float_dtype](K * N)
    C_device = ctx.enqueue_create_buffer[float_dtype](M * N)

    ctx.enqueue_copy(A_device, A_host)
    ctx.enqueue_copy(B_device, B_host)
    ctx.enqueue_copy(C_device, C_host)

    gpu_unsafe_naive_gemm(
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
        1)

    ctx.enqueue_copy(C_host, C_device)

    host_gemm[float_dtype](
        A_host.unsafe_ptr(),
        B_host.unsafe_ptr(),
        C_host2.unsafe_ptr(),
        M,
        N,
        K,
        K,
        N,
        N,
        1,
        1
    )

    ctx.synchronize()
    assert_ptr_equality[float_dtype](M * N, C_host.unsafe_ptr(), C_host2.unsafe_ptr())

def main():
    test_unsafe_naive_gemm()