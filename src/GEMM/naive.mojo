from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace
from sys import has_accelerator
from gpu.id import block_idx, thread_idx, block_dim
from math import ceildiv
from memory import UnsafePointer


fn _gpu_unsafe_naive_gemm[
    dtype: DType
](
    A_buffer: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GLOBAL, mut=False
    ],
    B_buffer: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GLOBAL, mut=False
    ],
    C_buffer: UnsafePointer[Scalar[dtype], address_space = AddressSpace.GLOBAL],
    M: UInt64,
    N: UInt64,
    K: UInt64,
    lda: UInt64,
    ldb: UInt64,
    ldc: UInt64,
    a: Scalar[dtype],
    b: Scalar[dtype],
):
    """
    Each thread is responsible for computing one elment it does
    this by iterating over the whole row of Matrix A and the whole
    column of matrix B. This version is made using UnsafePointers.
    """

    var C_column: UInt64 = thread_idx.x + block_idx.x * block_dim.x
    var C_row: UInt64 = thread_idx.y + block_idx.y * block_dim.y

    partial = Scalar[dtype](0)
    if C_row < M and C_column < N:
        for k in range(K):
            partial += A_buffer.load(C_row * lda + k) * B_buffer.load(
                k * ldb + C_column
            )

        scaled_c = C_buffer.load(C_row * N + C_column) * b + partial * a
        C_buffer.store(C_row * ldc + C_column, scaled_c)


fn gpu_unsafe_naive_gemm[
    dtype: DType
](
    ctx: DeviceContext,
    A_buffer: DeviceBuffer[dtype],
    B_Buffer: DeviceBuffer[dtype],
    C_Buffer: DeviceBuffer[dtype],
    M: UInt64,
    N: UInt64,
    K: UInt64,
    lda: UInt64,
    ldb: UInt64,
    ldc: UInt64,
    a: Scalar[dtype],
    b: Scalar[dtype],
) raises:

    ctx.enqueue_function[_gpu_unsafe_naive_gemm[dtype]](
        A_buffer,
        B_Buffer,
        C_Buffer,
        M,
        N,
        K,
        lda,
        ldb,
        ldc,
        a,
        b,
        grid_dim=(ceildiv(N, 16), ceildiv(M, 16)),
        block_dim=(16, 16),
    )
