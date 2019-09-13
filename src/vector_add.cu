#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>

#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

constexpr auto VECTOR_LENGTH = 1024u * 2;
constexpr auto EPS = 1e-4f;

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

void addHost(const std::vector<float> &A, const std::vector<float> &B,
        std::vector<float> &C) {
  assert(A.size() == B.size() && B.size() == C.size());
  for (auto i = 0u; i < A.size(); ++i) {
    C[i] = A[i] + B[i];
  }
}

__global__ void saxpy(const float *A, const float *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}

void addGPU(const std::vector<float> &A, const std::vector<float> &B,
        std::vector<float> &C) {
  float *A_gpu, *B_gpu, *C_gpu;
  auto byte_size = VECTOR_LENGTH * sizeof(float);
  GPU_CHECK(cudaMalloc(&A_gpu, byte_size));
  GPU_CHECK(cudaMalloc(&B_gpu, byte_size));
  GPU_CHECK(cudaMalloc(&C_gpu, byte_size));

  GPU_CHECK(cudaMemcpy(A_gpu, A.data(), byte_size, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(B_gpu, B.data(), byte_size, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(C_gpu, C.data(), byte_size, cudaMemcpyHostToDevice));

  saxpy<<<2, VECTOR_LENGTH / 2>>>(A_gpu, B_gpu, C_gpu);

  GPU_CHECK(cudaMemcpy(C.data(), C_gpu, byte_size, cudaMemcpyDeviceToHost));
}

bool verify(const std::vector<float> &A, const std::vector<float> &B,
            const std::vector<float> &C) {
  for (auto i = 0u; i < VECTOR_LENGTH; ++i) {
    if (A[i] + B[i] - C[i] > EPS) {
      std::cout << "ERROR! Index " << i << "\n";
      std::cout << A[i] + B[i] << "  " << C[i] << "\n";
      return false;
    }
  }
  return true;
}

int main() {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> dist{5, 2};

  std::vector<float> A(VECTOR_LENGTH), B(VECTOR_LENGTH), C(VECTOR_LENGTH);

  for (auto i = 0u; i < VECTOR_LENGTH; ++i) {
    A[i] = dist(gen);
    B[i] = dist(gen);
    C[i] = M_PI;
  }
  addHost(A, B, C);
  if (!verify(A, B, C)) {
    return 1;
  }
  std::cout << "Host verified\n";

  std::fill(C.begin(), C.end(), M_PI);

  addGPU(A, B, C);
  if (!verify(A, B, C)) {
    return 1;
  }
  std::cout << "GPU verified\n";

  return 0;
}
