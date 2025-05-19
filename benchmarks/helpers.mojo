from random import rand
from memory import UnsafePointer
from gpu.memory import AddressSpace
from testing import assert_equal


fn fill(
    ptr: UnsafePointer[Scalar[DType.float32]], 
    length: Int,
    min: Scalar[DType.float64] = -100,
    max: Scalar[DType.float64] = 100) raises:

    if (max < min):
        raise Error("invalid bounds provided to fill whole")

    rand[DType.float32](ptr, length, min=min, max=max)


fn host_gemm[
    dtype: DType
](
    A_buffer: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GENERIC, mut=False
    ],
    B_buffer: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GENERIC, mut=False
    ],
    C_buffer: UnsafePointer[Scalar[dtype], address_space = AddressSpace.GENERIC],
    M: UInt64,
    N: UInt64,
    K: UInt64,
    lda: UInt64,
    ldb: UInt64,
    ldc: UInt64,
    a: Scalar[dtype],
    b: Scalar[dtype],
):
    for i in range(M):
        for j in range(N):
            partial = Scalar[dtype](0)
            for k in range(K):
                partial += A_buffer[i * lda + k] * B_buffer[k * ldb + j]

            partial *= a

            C_buffer[i * ldc + j] *= b
            C_buffer[i * ldc + j] += partial



        

