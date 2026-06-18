#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

void launch_wmma_fp16(const __half* a,
                       const __half* b_col_major,
                       float* c,
                       int m,
                       int n,
                       int k,
                       cudaStream_t stream);

// For tf32 a and b are floats because tf32 starts from fp32 inputs
void launch_wmma_tf32(const float* a_row_major,
                      const float* b_col_major,
                      float* c_row_major,
                      int m,
                      int n,
                      int k,
                      cudaStream_t stream);
// A stream is like a line for gpu, a gpu can have many lines so you specify which line should you queue this function call in

namespace {

constexpr int kM = 4096;
constexpr int kN = 4096;
constexpr int kK = 11008;

// If we run out of memory it fails silenty so this is very important for debugging
inline void checkCuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    std::cerr << message << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

inline void checkCublas(cublasStatus_t status, const char* message) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << message << ": cuBLAS status " << static_cast<int>(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <typename T>
double max_abs_diff(const std::vector<T>& a, const std::vector<T>& b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result = std::max(result, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return result;
}

double relative_l2(const std::vector<float>& reference, const std::vector<float>& candidate) {
  long double numerator = 0.0L;
  long double denominator = 0.0L;
  for (size_t i = 0; i < reference.size(); ++i) {
    const long double ref = static_cast<long double>(reference[i]);
    const long double diff = ref - static_cast<long double>(candidate[i]);
    numerator += diff * diff;
    denominator += ref * ref;
  }
  return std::sqrt(static_cast<double>(numerator / std::max(denominator, 1.0L)));
}

double tflops(int m, int n, int k, float milliseconds) {
  const double seconds = static_cast<double>(milliseconds) / 1000.0;
  return (2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)) /
         (seconds * 1.0e12);
}

void fill_row_major(std::vector<float>& matrix, int rows, int cols, float scale, float bias) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const float x = static_cast<float>((row * 1315423911u) ^ (col * 2654435761u));
      const float s = std::sin((x + bias) * 0.0000001f);
      const float c = std::cos((x + bias) * 0.00000013f);
      matrix[static_cast<size_t>(row) * cols + col] = scale * (0.7f * s + 0.3f * c);
    }
  }
}

// We need these transpose methods because WMMA requires a in row-major and b in column-major
void transpose_row_major_to_col_major(const std::vector<float>& row_major,
                                      std::vector<float>& col_major,
                                      int rows,
                                      int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      col_major[static_cast<size_t>(col) * rows + row] = row_major[static_cast<size_t>(row) * cols + col];
    }
  }
}

void transpose_row_major_to_col_major_half(const std::vector<float>& row_major,
                                           std::vector<__half>& col_major,
                                           int rows,
                                           int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      col_major[static_cast<size_t>(col) * rows + row] = __float2half_rn(row_major[static_cast<size_t>(row) * cols + col]);
    }
  }
}

template <typename T>
// just copies from cpu -> gpu
void copy_to_device(T* dst, const std::vector<T>& src) {
  checkCuda(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D failed");
}

void time_cublas_sgemm(cublasHandle_t handle,
                       const float* d_a_row_major,
                       const float* d_b_row_major,
                       float* d_c_row_major,
                       int m,
                       int n,
                       int k,
                       float* elapsed_ms) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cudaEvent_t start{};
  cudaEvent_t stop{};
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed"); // Events are just stopwatches for the gpu processes
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  checkCuda(cudaEventRecord(start), "cudaEventRecord start failed");
  // Notice how we swap matrx "a" and matrix "b" because the function is natively hardwired to expect column-major arrays
  // So we swap them and use the property: (a x b)^T = a^T x b^T
  checkCublas(cublasSgemm(handle,
                          CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          &alpha,
                          d_b_row_major,
                          n,
                          d_a_row_major,
                          k,
                          &beta,
                          d_c_row_major,
                          n),
              "cublasSgemm failed");
  checkCuda(cudaEventRecord(stop), "cudaEventRecord stop failed");
  checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");
  checkCuda(cudaEventElapsedTime(elapsed_ms, start, stop), "cudaEventElapsedTime failed");

  checkCuda(cudaEventDestroy(start), "cudaEventDestroy start failed");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop failed");
}

} 

int main() {
  checkCuda(cudaSetDevice(0), "cudaSetDevice failed");

  cudaDeviceProp prop{};
  checkCuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties failed");
  std::cout << "Running on " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";

  if (prop.major < 7) {
    std::cerr << "This project requires Tensor Core-capable hardware.\n";
    return EXIT_FAILURE;
  }

  std::vector<float> h_a_row_major(static_cast<size_t>(kM) * kK);
  std::vector<float> h_b_row_major(static_cast<size_t>(kK) * kN);
  fill_row_major(h_a_row_major, kM, kK, 0.75f, 0.0f);
  fill_row_major(h_b_row_major, kK, kN, 0.50f, 1.0f);

  std::vector<float> h_b_col_major(static_cast<size_t>(kK) * kN);
  transpose_row_major_to_col_major(h_b_row_major, h_b_col_major, kK, kN);

  std::vector<__half> h_a_half_row_major(static_cast<size_t>(kM) * kK);
  std::vector<__half> h_b_half_col_major(static_cast<size_t>(kK) * kN);
  for (size_t i = 0; i < h_a_row_major.size(); ++i) {
    h_a_half_row_major[i] = __float2half_rn(h_a_row_major[i]);
  }
  transpose_row_major_to_col_major_half(h_b_row_major, h_b_half_col_major, kK, kN);

  std::vector<float> h_c_reference(static_cast<size_t>(kM) * kN, 0.0f);
  std::vector<float> h_c_tf32(static_cast<size_t>(kM) * kN, 0.0f);
  std::vector<float> h_c_fp16(static_cast<size_t>(kM) * kN, 0.0f);

  // Pointers for the matrix copies on the gpu
  float* d_a_row_major = nullptr;
  float* d_b_row_major = nullptr;
  float* d_b_col_major = nullptr;
  float* d_c_reference = nullptr;
  float* d_c_tf32 = nullptr;
  __half* d_a_half_row_major = nullptr;
  __half* d_b_half_col_major = nullptr;
  float* d_c_fp16 = nullptr;

  // Allocate memory on gpu
  checkCuda(cudaMalloc(&d_a_row_major, h_a_row_major.size() * sizeof(float)), "cudaMalloc A float failed");
  checkCuda(cudaMalloc(&d_b_row_major, h_b_row_major.size() * sizeof(float)), "cudaMalloc B float failed");
  checkCuda(cudaMalloc(&d_b_col_major, h_b_col_major.size() * sizeof(float)), "cudaMalloc B col float failed");
  checkCuda(cudaMalloc(&d_c_reference, h_c_reference.size() * sizeof(float)), "cudaMalloc C reference failed");
  checkCuda(cudaMalloc(&d_c_tf32, h_c_tf32.size() * sizeof(float)), "cudaMalloc C tf32 failed");

  // Copies data to the allocated memory on gpu
  copy_to_device(d_a_row_major, h_a_row_major);
  copy_to_device(d_b_row_major, h_b_row_major);
  copy_to_device(d_b_col_major, h_b_col_major);

  cublasHandle_t cublas{};
  checkCublas(cublasCreate(&cublas), "cublasCreate failed");
  checkCublas(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode failed");

  float cublas_ms = 0.0f;
  time_cublas_sgemm(cublas, d_a_row_major, d_b_row_major, d_c_reference, kM, kN, kK, &cublas_ms); // Our sgemm stopwatch fn
  checkCuda(cudaMemcpy(h_c_reference.data(), d_c_reference, h_c_reference.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy C reference D2H failed");

  cudaStream_t stream{};
  checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate failed");

  cudaEvent_t start{};
  cudaEvent_t stop{};
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  checkCuda(cudaMemsetAsync(d_c_tf32, 0, h_c_tf32.size() * sizeof(float), stream), "cudaMemsetAsync C tf32 failed");
  checkCuda(cudaEventRecord(start, stream), "cudaEventRecord start tf32 failed");
  launch_wmma_tf32(d_a_row_major, d_b_col_major, d_c_tf32, kM, kN, kK, stream);
  checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord stop tf32 failed");
  checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize tf32 failed");
  float tf32_ms = 0.0f;
  checkCuda(cudaEventElapsedTime(&tf32_ms, start, stop), "cudaEventElapsedTime tf32 failed");
  checkCuda(cudaMemcpy(h_c_tf32.data(), d_c_tf32, h_c_tf32.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy C tf32 D2H failed");

  checkCuda(cudaMalloc(&d_a_half_row_major, h_a_half_row_major.size() * sizeof(__half)), "cudaMalloc A half failed");
  checkCuda(cudaMalloc(&d_b_half_col_major, h_b_half_col_major.size() * sizeof(__half)), "cudaMalloc B half failed");
  checkCuda(cudaMalloc(&d_c_fp16, h_c_fp16.size() * sizeof(float)), "cudaMalloc C fp16 failed");
  copy_to_device(d_a_half_row_major, h_a_half_row_major);
  copy_to_device(d_b_half_col_major, h_b_half_col_major);

  checkCuda(cudaMemsetAsync(d_c_fp16, 0, h_c_fp16.size() * sizeof(float), stream), "cudaMemsetAsync C fp16 failed");
  checkCuda(cudaEventRecord(start, stream), "cudaEventRecord start fp16 failed");
  launch_wmma_fp16(d_a_half_row_major, d_b_half_col_major, d_c_fp16, kM, kN, kK, stream);
  checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord stop fp16 failed");
  checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize fp16 failed");
  float fp16_ms = 0.0f;
  checkCuda(cudaEventElapsedTime(&fp16_ms, start, stop), "cudaEventElapsedTime fp16 failed");
  checkCuda(cudaMemcpy(h_c_fp16.data(), d_c_fp16, h_c_fp16.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy C fp16 D2H failed");

  checkCuda(cudaEventDestroy(start), "cudaEventDestroy start failed");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop failed");
  checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy failed");

  checkCublas(cublasDestroy(cublas), "cublasDestroy failed");

  const double baseline_tflops = tflops(kM, kN, kK, cublas_ms);
  const double tf32_tflops = tflops(kM, kN, kK, tf32_ms);
  const double fp16_tflops = tflops(kM, kN, kK, fp16_ms);

  std::cout << "cuBLAS SGEMM: " << cublas_ms << " ms, " << baseline_tflops << " TFLOPS\n";
  std::cout << "WMMA TF32  : " << tf32_ms << " ms, " << tf32_tflops << " TFLOPS\n";
  std::cout << "WMMA FP16  : " << fp16_ms << " ms, " << fp16_tflops << " TFLOPS\n";

  std::cout << "TF32 max abs diff vs cuBLAS: " << max_abs_diff(h_c_reference, h_c_tf32)
            << ", relative L2: " << relative_l2(h_c_reference, h_c_tf32) << '\n';
  std::cout << "FP16 max abs diff vs cuBLAS: " << max_abs_diff(h_c_reference, h_c_fp16)
            << ", relative L2: " << relative_l2(h_c_reference, h_c_fp16) << '\n';

  cudaFree(d_a_row_major);
  cudaFree(d_b_row_major);
  cudaFree(d_b_col_major);
  cudaFree(d_c_reference);
  cudaFree(d_c_tf32);
  cudaFree(d_a_half_row_major);
  cudaFree(d_b_half_col_major);
  cudaFree(d_c_fp16);

  return EXIT_SUCCESS;
}
