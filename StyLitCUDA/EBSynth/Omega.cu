#include "Omega.cuh"

#include "../Utilities/Utilities.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {
namespace Omega {

__device__ void update(Image<float> &omegas, const int centerRow, const int centerCol,
                       float difference, int patchSize) {
  const int halfPatch = patchSize / 2;

  // Restricts the update to be within the omega array.
  const int rowLimit = omegas.rows - halfPatch;
  const int colLimit = omegas.cols - halfPatch;
  const int rowStart = Utilities::clamp(halfPatch, centerRow - halfPatch, rowLimit);
  const int colStart = Utilities::clamp(halfPatch, centerCol - halfPatch, colLimit);
  const int rowEnd = Utilities::clamp(halfPatch, centerRow + halfPatch + 1, rowLimit);
  const int colEnd = Utilities::clamp(halfPatch, centerCol + halfPatch + 1, colLimit);

  // Changes each value in the patch by difference.
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      atomicAdd(omegas.at(row, col), difference);
    }
  }
}

__global__ void zeroKernel(Image<float> omegas) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < omegas.rows && col < omegas.cols) {
    *omegas.at(row, col) = 0.f;
  }
}

__device__ float get(const Image<float> &omegas, const int centerRow, const int centerCol, const int patchSize) {
  const int halfPatch = patchSize / 2;

  // Restricts the update to be within the omega array.
  const int rowLimit = omegas.rows - halfPatch;
  const int colLimit = omegas.cols - halfPatch;
  const int rowStart = Utilities::clamp(halfPatch, centerRow - halfPatch, rowLimit);
  const int colStart = Utilities::clamp(halfPatch, centerCol - halfPatch, colLimit);
  const int rowEnd = Utilities::clamp(halfPatch, centerRow + halfPatch + 1, rowLimit);
  const int colEnd = Utilities::clamp(halfPatch, centerCol + halfPatch + 1, colLimit);

  // Changes each value in the patch by difference.
  float result = 0.f;
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      result += *omegas.constAt(row, col);
    }
  }
  return result;
}

__global__ void initializeKernel(const Image<NNFEntry> nnf, Image<float> omegas,
                                 const int patchSize) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < nnf.rows && col < nnf.cols) {
    const NNFEntry *entry = nnf.constAt(row, col);
    update(omegas, entry->row, entry->col, 1, patchSize);
  }
}

void initialize(const Image<NNFEntry> &nnf, Image<float> &omegas, int patchSize) {
  printf("EBSynthCUDA: Constructing omega array with dimensions [%d, %d] using NNF with dimensions "
         "[%d, %d].\n",
         omegas.rows, omegas.cols, nnf.rows, nnf.cols);

  // Calculates the block size.
  const int BLOCK_SIZE_2D = 16;
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 perNNF(Utilities::divideRoundUp(nnf.rows, threadsPerBlock.x),
                    Utilities::divideRoundUp(nnf.cols, threadsPerBlock.y));
  const dim3 perOmega(Utilities::divideRoundUp(omegas.rows, threadsPerBlock.x),
                      Utilities::divideRoundUp(omegas.cols, threadsPerBlock.y));

  // Runs the kernel that randomizes NNF entries.
  zeroKernel<<<perOmega, threadsPerBlock>>>(omegas);
  check(cudaDeviceSynchronize());
  initializeKernel<<<perNNF, threadsPerBlock>>>(nnf, omegas, patchSize);
  check(cudaDeviceSynchronize());
}

} /* namespace Omega */
} /* namespace StyLitCUDA */
