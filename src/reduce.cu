#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <limits>
#include <vector>
#include <chrono>

constexpr auto VECTOR_LENGTH = 1024u * 1024u * 16u;
constexpr auto EPS = 1e-6f;

#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << ":"
              << line << "\n";
    if (abort) {
      std::exit(code);
    }
  }
}

float findMaxHost(const std::vector<float> &A) {
  auto time1 = std::chrono::steady_clock::now();
  auto it = std::max_element(std::begin(A), std::end(A));
  auto time2 = std::chrono::steady_clock::now();
  std::cout << "CPU: "
            << std::chrono::duration_cast<std::chrono::microseconds>(time2 -
                                                                     time1)
                   .count()
            << "\n";
  return *it;
}

constexpr auto ELEMENTS_PER_WORKITEM = 8u;
constexpr auto WORKGROUP_SIZE = 128u;
constexpr auto ELEMENTS_PER_BLOCK = WORKGROUP_SIZE * ELEMENTS_PER_WORKITEM;
constexpr auto MIN_FLOAT = std::numeric_limits<float>::min();

__device__ void warpReduce(volatile float *shared, int tid) {
  shared[tid] = fmaxf(shared[tid], shared[tid + 32]);
  shared[tid] = fmaxf(shared[tid], shared[tid + 16]);
  shared[tid] = fmaxf(shared[tid], shared[tid + 8]);
  shared[tid] = fmaxf(shared[tid], shared[tid + 4]);
  shared[tid] = fmaxf(shared[tid], shared[tid + 2]);
  shared[tid] = fmaxf(shared[tid], shared[tid + 1]);
}

__global__ void maxKernel(float *A, float *result, int N) {
  extern __shared__ float shared[];

  int i = blockIdx.x * blockDim.x * ELEMENTS_PER_WORKITEM + threadIdx.x;
  float max = MIN_FLOAT;

  for (int j = 0; j < ELEMENTS_PER_WORKITEM; ++j) {
    i += blockDim.x;
    if (i < N) {
      max = fmaxf(max, A[i]);
    }
  }

  shared[threadIdx.x] = max;
  __syncthreads();

  for (int max_thread_id = blockDim.x / 2; max_thread_id > 32;
       max_thread_id /= 2) {
    if (threadIdx.x < max_thread_id) {
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + max_thread_id]);
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    warpReduce(shared, threadIdx.x);
  }
  if (threadIdx.x == 0) {
    result[blockIdx.x] = shared[0];
  }
}

float findMaxGPU(const std::vector<float> &A) {
  float *A_gpu, *temp_gpu;
  auto byte_size = VECTOR_LENGTH * sizeof(float);
  GPU_CHECK(cudaMalloc(&A_gpu, byte_size));
  GPU_CHECK(cudaMemcpy(A_gpu, A.data(), byte_size, cudaMemcpyHostToDevice));

  auto block_count = VECTOR_LENGTH / ELEMENTS_PER_BLOCK;
  GPU_CHECK(cudaMalloc(&temp_gpu, block_count * sizeof(float)));
  GPU_CHECK(cudaDeviceSynchronize());

  auto time1 = std::chrono::steady_clock::now();
  maxKernel<<<block_count, WORKGROUP_SIZE, WORKGROUP_SIZE * sizeof(float)>>>(
      A_gpu, temp_gpu, VECTOR_LENGTH);
  GPU_CHECK(cudaDeviceSynchronize());
  auto time2 = std::chrono::steady_clock::now();

  std::vector<float> temp_host(block_count);
  GPU_CHECK(cudaMemcpy(temp_host.data(), temp_gpu, block_count * sizeof(float),
                       cudaMemcpyDeviceToHost));
  auto it = std::max_element(std::begin(temp_host), std::end(temp_host));

  GPU_CHECK(cudaFree(A_gpu));
  GPU_CHECK(cudaFree(temp_gpu));
  std::cout << "GPU: "
            << std::chrono::duration_cast<std::chrono::microseconds>(time2 -
                                                                     time1)
                   .count()
            << "\n";
  return *it;
}

int main() {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> dist{5, 2};

  std::vector<float> A(VECTOR_LENGTH);

  for (auto i = 0u; i < VECTOR_LENGTH; ++i) {
    A[i] = dist(gen);
  }

  auto max_host = findMaxHost(A);
  auto max_device = findMaxGPU(A);

  if (std::abs(max_host - max_device) > EPS) {
    std::cout << "ERROR\n";
    std::cout << max_host << " : " << max_device << "\n";
    return 1;
  } else {
    std::cout << "SUCCESS\n";
  }

  return 0;
}

