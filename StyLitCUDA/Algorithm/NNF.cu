#include "NNF.cuh"

#include "../Utilities/Coordinates.cuh"
#include "../Utilities/Utilities.cuh"
#include "Error.cuh"
#include "PCG.cuh"

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {
namespace NNF {

template <typename T>
__global__ void randomizeKernel(Image<NNFEntry> nnf, Image<PCGState> random, const Image<T> from,
                                const Image<T> to, const int patchSize) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < nnf.rows && col < nnf.cols) {
    // Generates a random mapping.
    const int mappedRow = PCG::rand(random.at(row, col)) % to.rows;
    const int mappedCol = PCG::rand(random.at(row, col)) % to.cols;

    // Fills the random mapping in.
    NNFEntry *entry = nnf.at(row, col);
    entry->row = mappedRow;
    entry->col = mappedCol;
    entry->error = Error::calculate(from, to, Coordinates(row, col),
                                    Coordinates(mappedRow, mappedCol), patchSize);
  }
}

template <typename T>
void randomize(Image<NNFEntry> &nnf, Image<PCGState> &random, const Image<T> &from,
               const Image<T> &to, const int patchSize) {
  assert(nnf.rows <= random.rows && nnf.cols <= random.cols);
  printf("StyLitCUDA: Randomly initializing NNF with dimensions [%d, %d] using array of random "
         "states with dimensions [%d, %d].\n",
         nnf.rows, nnf.cols, random.rows, random.cols);

  // Calculates the block size.
  const int BLOCK_SIZE_2D = 16;
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 numBlocks(Utilities::divideRoundUp(nnf.rows, threadsPerBlock.x),
                       Utilities::divideRoundUp(nnf.cols, threadsPerBlock.y));

  // Runs the kernel that randomizes NNF entries.
  randomizeKernel<<<numBlocks, threadsPerBlock>>>(nnf, random, from, to, patchSize);
  check(cudaDeviceSynchronize());
}

template void randomize(Image<NNFEntry> &nnf, Image<PCGState> &random, const Image<int> &from,
                        const Image<int> &to, const int patchSize);
template void randomize(Image<NNFEntry> &nnf, Image<PCGState> &random, const Image<float> &from,
                        const Image<float> &to, const int patchSize);

} /* namespace NNF */
} /* namespace StyLitCUDA */
