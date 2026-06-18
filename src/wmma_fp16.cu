#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace {

__global__ void wmma_fp16_kernel(const __half* a_row_major,
                                 const __half* b_col_major,
                                 float* c_row_major,
                                 int m,
                                 int n,
                                 int k) {
  constexpr int tile_m = 16;
  constexpr int tile_n = 16;
  constexpr int tile_k = 16;

  const int row = blockIdx.y * tile_m;
  const int col = blockIdx.x * tile_n;

  if (row >= m || col >= n) {
    return;
  }

  wmma::fragment<wmma::matrix_a, tile_m, tile_n, tile_k, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, tile_m, tile_n, tile_k, __half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, tile_m, tile_n, tile_k, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int kk = 0; kk < k; kk += tile_k) {
    const __half* a_tile = a_row_major + static_cast<size_t>(row) * k + kk;
    const __half* b_tile = b_col_major + kk + static_cast<size_t>(col) * k;

    wmma::load_matrix_sync(a_frag, a_tile, k);
    wmma::load_matrix_sync(b_frag, b_tile, k);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  float* c_tile = c_row_major + static_cast<size_t>(row) * n + col;
  wmma::store_matrix_sync(c_tile, c_frag, n, wmma::mem_row_major);
}

} // namespace

void launch_wmma_fp16(const __half* a_row_major,
                      const __half* b_col_major,
                      float* c_row_major,
                      int m,
                      int n,
                      int k,
                      cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n + 15) / 16, (m + 15) / 16);
  wmma_fp16_kernel<<<grid, block, 0, stream>>>(a_row_major, b_col_major, c_row_major, m, n, k);
}
