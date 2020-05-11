#include "RandomInitializer.cuh"

#include "../Utilities/Utilities.cuh"
#include "PCG.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {

__global__ void initializeRandomStateKernel(Image<PCGState> random, const int seed) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < random.rows && col < random.cols) {
    PCG::init(random.at(row, col), seed, row * random.cols + col);
  }
}

void initializeRandomState(const Image<PCGState> &random) {
  printf("StyLitCUDA: Initializing PCG state with dimensions [%d, %d].\n", random.rows,
         random.cols);

  // Calculates the block size.
  const int BLOCK_SIZE_2D = 16;
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 numBlocks(Utilities::divideRoundUp(random.rows, threadsPerBlock.x),
                       Utilities::divideRoundUp(random.cols, threadsPerBlock.y));

  // Runs the kernel that initializes PCG states everywhere in the image.
  initializeRandomStateKernel<<<numBlocks, threadsPerBlock>>>(random, 503);
  check(cudaDeviceSynchronize());
}

} /* namespace StyLitCUDA */
