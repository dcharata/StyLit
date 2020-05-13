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
                                const Image<T> to, const int patchSize, const Vec<float> guideWeights, const Vec<float> styleWeights) {
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
                                    Coordinates(mappedRow, mappedCol), patchSize, guideWeights, styleWeights);
  }
}

template <typename T>
void randomize(Image<NNFEntry> &nnf, Image<PCGState> &random, const Image<T> &from,
               const Image<T> &to, const int patchSize, const Vec<float> &guideWeights, const Vec<float> &styleWeights) {
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
  randomizeKernel<<<numBlocks, threadsPerBlock>>>(nnf, random, from, to, patchSize, guideWeights, styleWeights);
  check(cudaDeviceSynchronize());
}

template void randomize(Image<NNFEntry> &nnf, Image<PCGState> &random, const Image<int> &from,
                        const Image<int> &to, const int patchSize, const Vec<float> &guideWeights, const Vec<float> &styleWeights);
template void randomize(Image<NNFEntry> &nnf, Image<PCGState> &random, const Image<float> &from,
                        const Image<float> &to, const int patchSize, const Vec<float> &guideWeights, const Vec<float> &styleWeights);

template <typename T>
__global__ void recalculateErrorsKernel(Image<NNFEntry> nnf, const Image<T> from, const Image<T> to,
                                        const int patchSize, const Vec<float> guideWeights, const Vec<float> styleWeights) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < nnf.rows && col < nnf.cols) {
    NNFEntry *entry = nnf.at(row, col);
    entry->error = Error::calculate(from, to, Coordinates(row, col),
                                    Coordinates(entry->row, entry->col), patchSize, guideWeights, styleWeights);
  }
}

template <typename T>
void recalculateErrors(Image<NNFEntry> &nnf, const Image<T> &from, const Image<T> &to,
                       const int patchSize, const Vec<float> &guideWeights, const Vec<float> &styleWeights) {
  printf("StyLitCUDA: Recalculating errors for NNF with dimensions [%d, %d].\n", nnf.rows,
         nnf.cols);

  // Calculates the block size.
  const int BLOCK_SIZE_2D = 16;
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 numBlocks(Utilities::divideRoundUp(nnf.rows, threadsPerBlock.x),
                       Utilities::divideRoundUp(nnf.cols, threadsPerBlock.y));

  // Runs the kernel that recalculates NNF error entries.
  recalculateErrorsKernel<<<numBlocks, threadsPerBlock>>>(nnf, from, to, patchSize, guideWeights, styleWeights);
  check(cudaDeviceSynchronize());
}

template void recalculateErrors(Image<NNFEntry> &nnf, const Image<int> &from, const Image<int> &to,
                                const int patchSize, const Vec<float> &guideWeights, const Vec<float> &styleWeights);
template void recalculateErrors(Image<NNFEntry> &nnf, const Image<float> &from,
                                const Image<float> &to, const int patchSize, const Vec<float> &guideWeights, const Vec<float> &styleWeights);

__global__ void upscaleKernel(const Image<NNFEntry> from, Image<NNFEntry> to, const int patchSize) {
  const int toRow = blockDim.x * blockIdx.x + threadIdx.x;
  const int toCol = blockDim.y * blockIdx.y + threadIdx.y;
  if (toRow < to.rows && toCol < to.cols) {
    const int fromRow = Utilities::clamp(0, toRow / 2, from.rows);
    const int fromCol = Utilities::clamp(0, toCol / 2, from.cols);
    const NNFEntry *fromEntry = from.constAt(fromRow, fromCol);
    NNFEntry *toEntry = to.at(toRow, toCol);
    const int halfPatch = patchSize / 2;
    toEntry->row =
        Utilities::clamp(halfPatch, fromEntry->row * 2 + (toRow % 2), to.rows - halfPatch);
    toEntry->col =
        Utilities::clamp(halfPatch, fromEntry->col * 2 + (toCol % 2), to.cols - halfPatch);
    toEntry->error = fromEntry->error;
  }
}

void upscale(const Image<NNFEntry> &from, Image<NNFEntry> &to, const int patchSize) {
  assert(to.rows / 2 == from.rows && to.cols / 2 == from.cols);
  printf("StyLitCUDA: Upscaling NNF with dimensions [%d, %d] to dimensions [%d, %d].\n", from.rows,
         from.cols, to.rows, to.cols);

  // Calculates the block size.
  const int BLOCK_SIZE_2D = 16;
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 numBlocks(Utilities::divideRoundUp(to.rows, threadsPerBlock.x),
                       Utilities::divideRoundUp(to.cols, threadsPerBlock.y));

  // Runs the kernel that randomizes NNF entries.
  upscaleKernel<<<numBlocks, threadsPerBlock>>>(from, to, patchSize);
  check(cudaDeviceSynchronize());
}

__global__ void invalidateKernel(Image<NNFEntry> nnf) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < nnf.rows && col < nnf.cols) {
    NNFEntry *entry = nnf.at(row, col);
    entry->row = NNF::INVALID;
    entry->col = NNF::INVALID;
    entry->error = 0.f;
  }
}

void invalidate(Image<NNFEntry> &nnf) {
  printf("StyLitCUDA: Invalidating NNF with dimensions [%d, %d].\n", nnf.rows, nnf.cols);

  // Calculates the block size.
  const int BLOCK_SIZE_2D = 16;
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 numBlocks(Utilities::divideRoundUp(nnf.rows, threadsPerBlock.x),
                       Utilities::divideRoundUp(nnf.cols, threadsPerBlock.y));

  // Runs the kernel that randomizes NNF entries.
  invalidateKernel<<<numBlocks, threadsPerBlock>>>(nnf);
  check(cudaDeviceSynchronize());
}

} /* namespace NNF */
} /* namespace StyLitCUDA */
