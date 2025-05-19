from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace
from sys import has_accelerator
from gpu import block_idx, thread_idx, block_dim, barrier
from math import ceildiv
from memory import UnsafePointer, stack_allocation
from layout import Layout, LayoutTensor


# TODO Test and add TILE iteration one using runtime TILE layout and benchmark then move on 

@always_inline
fn load_to_shared_memory[
    dtype: DType, 
    A_layout: Layout, 
    B_layout: Layout, 
    TILE_SIZE_X: Int, 
    TILE_SIZE_Y: Int, 
    TILE_SIZE_K: Int,
    THREADS_PER_BLOCK: Int](
    A_buffer: UnsafePointer[Scalar[dtype], address_space = AddressSpace.GLOBAL, mut=False],
    B_buffer: UnsafePointer[Scalar[dtype], address_space = AddressSpace.GLOBAL, mut=False],
    A_thread_block_tile: LayoutTensor[dtype, A_layout, address_space = AddressSpace.SHARED],
    B_thread_block_tile: LayoutTensor[dtype, B_layout, address_space = AddressSpace.SHARED],
    M: UInt64,
    N: UInt64,
    K: UInt64,
    thread_linear_idx: UInt64,
    lda: UInt64,
    ldb: UInt64,
    iteration: Int
):
    var A_block_iters = ceildiv(THREADS_PER_BLOCK, A_layout.size())
    var A_iterations = A_thread_block_tile.tiled_iterator[TILE_SIZE_X, TILE_SIZE_Y, axis=0](0, 0)

    for a_iter in range(A_block_iters):

        a_tile = A_iterations.get()

        var temp = Scalar[dtype](0)

        var A_block_row = thread_linear_idx + (THREADS_PER_BLOCK * a_iter) // TILE_SIZE_K 
        var A_block_column = thread_linear_idx + (THREADS_PER_BLOCK * a_iter) % TILE_SIZE_K

        var A_row = block_idx.y * TILE_SIZE_Y + A_block_row
        var A_column = (iteration * TILE_SIZE_K) + A_block_column

        if (A_column < K and A_row < M):
            temp = A_buffer[A_row * lda + A_column]

        a_tile[thread_idx.y][thread_idx.x] = temp
        _ = A_iterations.next() # TODO may be a bug

    
    var B_block_iters = ceildiv(THREADS_PER_BLOCK, B_layout.size())
    var B_iterations = B_thread_block_tile.tiled_iterator[TILE_SIZE_X, TILE_SIZE_Y, axis=0](0, 0)

    for b_iter in range(B_block_iters):

        b_tile = B_iterations.get()

        var temp = Scalar[dtype](0)

        var B_block_row = thread_linear_idx + (THREADS_PER_BLOCK * b_iter) // TILE_SIZE_X 
        var B_block_column = thread_linear_idx + (THREADS_PER_BLOCK * b_iter) % TILE_SIZE_X

        var B_row = iteration * TILE_SIZE_K + B_block_row
        var B_column = B_block_column + (TILE_SIZE_X * block_dim.x)

        if (B_row < K and B_column < N):
            temp = B_buffer[B_row * ldb + B_column]

        b_tile[thread_idx.y][thread_idx.x] = temp
        _ = B_iterations.next() # TODO may be a bug




fn _gpu_2D_block_tiling[
    dtype: DType,
    A_layout: Layout,
    B_layout: Layout
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

    alias TILE_SIZE_Y = A_layout.shape.value(0)
    alias TILE_SIZE_K = A_layout.shape.value(1)
    alias TILE_SIZE_X = B_layout.shape.value(0)

    alias THREADS_PER_BLOCK = TILE_SIZE_X * TILE_SIZE_Y

    var A_shared_buffer = stack_allocation[
        A_layout.size(),
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()

    var B_shared_buffer = stack_allocation[
        B_layout.size(),
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()

    var A_thread_block_tile = LayoutTensor[dtype, A_layout, address_space = AddressSpace.SHARED](A_shared_buffer)
    var B_thread_block_tile = LayoutTensor[dtype, B_layout, address_space = AddressSpace.SHARED](B_shared_buffer)

    total_iters = ceildiv(K, TILE_SIZE_K)

    var thread_linear_idx = thread_idx.y * TILE_SIZE_X + thread_idx.x

    var C_col: UInt64 = TILE_SIZE_X * block_idx.x + thread_idx.x
    var C_row: UInt64 = TILE_SIZE_Y * block_idx.y + thread_idx.y

    var partial = Scalar[dtype](0)
    for iteration in range(total_iters):

        load_to_shared_memory[
            dtype,
            A_layout,
            B_layout,
            TILE_SIZE_X,
            TILE_SIZE_Y,
            TILE_SIZE_K,
            THREADS_PER_BLOCK
        ](
            A_buffer,
            B_buffer,
            A_thread_block_tile,
            B_thread_block_tile,
            M,
            N,
            K,
            thread_linear_idx,
            lda,
            ldb,
            Int(iteration)
        )
        
        barrier()

        for k in range(TILE_SIZE_K):
            partial += A_thread_block_tile[thread_idx.y][k] * B_thread_block_tile[k][thread_idx.x]

        barrier()

    if C_row < M and C_col < N:
        C_buffer[C_row * ldc + C_col] = C_buffer[C_row * ldc + C_col] * b + partial * a


    


