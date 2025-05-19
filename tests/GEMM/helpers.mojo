from random import random_si64
from memory import UnsafePointer
from gpu.memory import AddressSpace
from testing import assert_equal


fn fill_whole(
    ptr: UnsafePointer[Scalar[DType.float32]], 
    length: Scalar[DType.uint64],
    min: Scalar[DType.int64] = -100,
    max: Scalar[DType.int64] = 100) raises:

    if (max < min):
        raise Error("invalid bounds provided to fill whole")
    
    for i in range(length):
        ptr[i] = Float32(random_si64(min, max))


fn assert_ptr_equality[dtype: DType](
    total_items: Scalar[DType.uint64],
    first: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GENERIC, mut=False
    ],
    second: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GENERIC, mut=False
    ],
) raises:
    for i in range(total_items):
        assert_equal(first[i], second[i])



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



        

