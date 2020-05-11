#include "PatchMatch.cuh"

#include "../Utilities/Coordinates.cuh"
#include "../Utilities/Utilities.cuh"
#include "Error.cuh"
#include "PCG.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {
namespace PatchMatch {

/**
 * @brief tryPatch Calculates the error for a new mapping from [sourceRow, sourceCol] to [targetRow,
 * targetCol]. If this error is lower than the error for the existing mapping at [sourceRow,
 * sourceCol] in previousNNF, replaces the existing mapping at [sourceRow, sourceCol] with the new
 * mapping.
 * @param source the source image
 * @param target the target image
 * @param nextNNF the NNF to modify with improved mappings
 * @param previousNNF the NNF to read from
 * @param patchSize the patch width/height
 * @param sourceRow the source row to try a mapping for
 * @param sourceCol the source column to try a mapping for
 * @param targetRow the target row to try a mapping for
 * @param targetCol the target column to try a mapping for
 */
template <typename T>
__device__ void tryPatch(const Image<T> &source, const Image<T> &target, Image<NNFEntry> &nextNNF,
                         const Image<NNFEntry> &previousNNF, const int patchSize,
                         const int sourceRow, const int sourceCol, const int targetRow,
                         const int targetCol) {
  // Calculates the error for the new mapping and compares it with the existing error.
  const float oldError = previousNNF.constAt(sourceRow, sourceCol)->error;
  const float newError = Error::calculate(source, target, Coordinates(sourceRow, sourceCol),
                                          Coordinates(targetRow, targetCol), patchSize);
  if (newError <= oldError) {
    // If the new error would be lower, updates nextNNF.
    NNFEntry *nextMapping = nextNNF.at(sourceRow, sourceCol);
    nextMapping->row = targetRow;
    nextMapping->col = targetCol;
    nextMapping->error = newError;
  }
}

/**
 * @brief tryNeighborOffset Replaces a mapping with a neighboring mapping (but offset) if doing
 * so reduces the overall error. See propagationPassKernel for more details.
 * @param source the source image
 * @param target the target image
 * @param nextNNF the NNF to modify with improved mappings
 * @param previousNNF the NNF to read from
 * @param patchSize the patch width/height
 * @param sourceRow the source row to try an offset for
 * @param sourceCol the source column to try an offset for
 * @param rowOffset the row offset to try
 * @param colOffset the column offset to try
 */
template <typename T>
__device__ void tryNeighborOffset(const Image<T> &source, const Image<T> &target,
                                  Image<NNFEntry> &nextNNF, const Image<NNFEntry> &previousNNF,
                                  const int patchSize, const int sourceRow, const int sourceCol,
                                  const int rowOffset, const int colOffset) {
  // Gets the neighbor's mapping.
  const int neighborRow = Utilities::clamp(0, sourceRow + rowOffset, source.rows);
  const int neighborCol = Utilities::clamp(0, sourceCol + colOffset, source.cols);
  const NNFEntry &neighborMapping = *previousNNF.constAt(neighborRow, neighborCol);

  // Translates the neighbor's mapping back to get the target coordinates to try.
  const int targetRow = Utilities::clamp(0, neighborMapping.row - rowOffset, target.rows);
  const int targetCol = Utilities::clamp(0, neighborMapping.col - colOffset, target.cols);

  // Tries the mapping.
  tryPatch(source, target, nextNNF, previousNNF, patchSize, sourceRow, sourceCol, targetRow,
           targetCol);
}

/**
 * @brief propagationPassKernel This compares each pixel to neighboring pixels to see if their
 * mappings, when shifted to the current pixel, would be better than the current mapping. This
 * makes good mappings propagate to nearby pixels.
 * @param source the source image
 * @param target the target image
 * @param nextNNF the NNF to modify with improved mappings
 * @param previousNNF the NNF to read from
 * @param patchSize the patch width/height
 * @param offset the offset to try in each direction
 */
template <typename T>
__global__ void propagationPassKernel(const Image<T> source, const Image<T> target,
                                      Image<NNFEntry> nextNNF, const Image<NNFEntry> previousNNF,
                                      const int patchSize, const int offset) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < source.rows && col < source.cols) {
    // Tries offsetting up, down, left and right.
    tryNeighborOffset(source, target, nextNNF, previousNNF, patchSize, row, col, offset, 0);
    tryNeighborOffset(source, target, nextNNF, previousNNF, patchSize, row, col, -offset, 0);
    tryNeighborOffset(source, target, nextNNF, previousNNF, patchSize, row, col, 0, offset);
    tryNeighborOffset(source, target, nextNNF, previousNNF, patchSize, row, col, 0, -offset);
  }
}

/**
 * @brief randomSearchPassKernel Attempts to improve the NNF by randomly shifting its mapping (i.e.
 * the target coordinates) within a region with the specified radius. If the randomly shifted error
 * is better, the mapping is updated in nextNNF.
 * @param source the source image
 * @param target the target image
 * @param nextNNF the NNF to modify with improved mappings
 * @param previousNNF the NNF to read from
 * @param patchSize the patch width/height
 * @param radius the radius to randomly search within
 * @param random the pseudorandom number generator's state
 */
template <typename T>
__global__ void randomSearchPassKernel(const Image<T> source, const Image<T> target,
                                       Image<NNFEntry> nextNNF, const Image<NNFEntry> previousNNF,
                                       const int patchSize, const int radius,
                                       Image<PCGState> random) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < source.rows && col < source.cols) {
    // Gets the current mapping.
    const NNFEntry *previousMapping = previousNNF.constAt(row, col);

    // Randomly shifts the mapping within the radius.
    PCGState *randomState = random.at(row, col);
    const int range = 2 * radius;
    const int rowShift = PCG::rand(randomState) % range - radius;
    const int colShift = PCG::rand(randomState) % range - radius;
    const int newTargetRow = Utilities::clamp(0, previousMapping->row + rowShift, target.rows);
    const int newTargetCol = Utilities::clamp(0, previousMapping->col + colShift, target.cols);

    // Tries the shifted patch.
    tryPatch(source, target, nextNNF, previousNNF, patchSize, row, col, newTargetRow, newTargetCol);
  }
}

/**
 * @brief swapNNFs Swaps the pointers. This is used to alternate which NNF is read from and which
 * NNF is written to.
 * @param a one of the NNFs
 * @param b the other NNF
 */
void swapNNFs(Image<NNFEntry> **a, Image<NNFEntry> **b) {
  Image<NNFEntry> *temp = *a;
  *a = *b;
  *b = temp;
}

/**
 * @brief totalError Calculates the sum of the given NNF's errors.
 * @param nnf the NNF to calculate the error sum for
 * @return the sum of the NNF's errors
 */
float totalError(const Image<NNFEntry> &nnf) {
  // Copies the NNF to host memory.
  const int hostPitch = nnf.cols * sizeof(NNFEntry);
  const int nnfSizeInBytes = nnf.rows * hostPitch;
  NNFEntry *hostNNF;
  check(cudaMallocHost(&hostNNF, nnfSizeInBytes));
  check(cudaMemcpy2D((void *)hostNNF, hostPitch, nnf.deviceData, nnf.pitch, hostPitch, nnf.rows,
                     cudaMemcpyDeviceToHost));

  // Counts the sum of the errors.
  float error = 0.f;
  for (int row = 0; row < nnf.rows; row++) {
    for (int col = 0; col < nnf.cols; col++) {
      error += hostNNF[row * nnf.cols + col].error;
    }
  }
  check(cudaFreeHost(hostNNF));
  return error;
}

template <typename T>
void run(Image<NNFEntry> &nnf, const Image<NNFEntry> *blacklist, const Image<T> &from,
         const Image<T> &to, const Image<PCGState> &random, const int patchSize,
         const int numIterations) {
  printf("StyLitCUDA: Running PatchMatch with %d iterations, patch size %d, and %s blacklist.\n",
         numIterations, patchSize, blacklist ? "a" : "no");
  const float initialError = totalError(nnf);

  // Calculates the block size.
  const int BLOCK_SIZE_2D = 16;
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 numBlocks(Utilities::divideRoundUp(from.rows, threadsPerBlock.x),
                       Utilities::divideRoundUp(from.cols, threadsPerBlock.y));

  // Creates a temporary NNF that's used for alternating read/write.
  Image<NNFEntry> tempNNF(nnf.rows, nnf.cols, 1);
  tempNNF.allocate();

  // Runs iterations of PatchMatch.
  Image<NNFEntry> *previousNNF = &nnf;
  Image<NNFEntry> *nextNNF = &tempNNF;
  for (int iteration = 0; iteration < 1; iteration++) {
    // First, runs three propagation passes with offsets of 4, 2 and 1 respectively.
    propagationPassKernel<T>
        <<<numBlocks, threadsPerBlock>>>(from, to, *nextNNF, *previousNNF, patchSize, 4);
    check(cudaDeviceSynchronize());
    swapNNFs(&previousNNF, &nextNNF);

    propagationPassKernel<T>
        <<<numBlocks, threadsPerBlock>>>(from, to, *nextNNF, *previousNNF, patchSize, 2);
    check(cudaDeviceSynchronize());
    swapNNFs(&previousNNF, &nextNNF);

    propagationPassKernel<T>
        <<<numBlocks, threadsPerBlock>>>(from, to, *nextNNF, *previousNNF, patchSize, 1);
    check(cudaDeviceSynchronize());
    swapNNFs(&previousNNF, &nextNNF);

    // Next, runs a number of random search passes.
    for (int radius = std::max(from.rows, from.cols) / 2; radius > 1; radius /= 2) {
      randomSearchPassKernel<T><<<numBlocks, threadsPerBlock>>>(from, to, *nextNNF, *previousNNF,
                                                                patchSize, radius, random);
      check(cudaDeviceSynchronize());
      swapNNFs(&previousNNF, &nextNNF);
    }
  }

  // Since every NNF-altering call has a call to swapNNFs after it, previousNNF is actually the
  // most up-to-date NNF. It's swapped back here to make the code below less confusing.
  swapNNFs(&previousNNF, &nextNNF);

  // Copies the data from nextNNF to the input NNF if they aren't the same.
  if (nextNNF != &nnf) {
    check(cudaMemcpy2D(nnf.deviceData, nnf.pitch, nextNNF->deviceData, nextNNF->pitch,
                       nnf.cols * sizeof(NNFEntry), nnf.rows, cudaMemcpyDeviceToDevice));
  }
  printf("StyLitCUDA: PatchMatch reduced error from %f to %f.\n", initialError, totalError(nnf));

  // Frees the temporary NNF.
  tempNNF.free();
}

template void run(Image<NNFEntry> &nnf, const Image<NNFEntry> *blacklist, const Image<int> &from,
                  const Image<int> &to, const Image<PCGState> &random, const int patchSize,
                  const int numIterations);

template void run(Image<NNFEntry> &nnf, const Image<NNFEntry> *blacklist, const Image<float> &from,
                  const Image<float> &to, const Image<PCGState> &random, const int patchSize,
                  const int numIterations);

} /* namespace PatchMatch */
} /* namespace StyLitCUDA */
