#include "EBSynthPatchMatch.cuh"

#include "../Algorithm/NNF.cuh"
#include "../Algorithm/PCG.cuh"
#include "../Utilities/Coordinates.cuh"
#include "../Utilities/Utilities.cuh"
#include "../Algorithm/Error.cuh"
#include "Omega.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <limits>
#include <stdio.h>

namespace StyLitCUDA {
namespace EBSynthPatchMatch {

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
 * @param optionalBlacklist An optional blacklist (NNF that goes in the other direction). If a
 * location in the blacklist has a valid mapping, it should not be mapped to. A blacklist with zero
 * dimensions is ignored.
 */
template <typename T>
__device__ void tryPatch(const Image<T> &source, const Image<T> &target, Image<NNFEntry> &nextNNF,
                         const Image<NNFEntry> &previousNNF, const int patchSize,
                         const int sourceRow, const int sourceCol, const int targetRow,
                         const int targetCol, Image<float> &omegas, const float *weights) {
  // Calculates the error for the new mapping and compares it with the existing error.
  const NNFEntry *previousMapping = previousNNF.constAt(sourceRow, sourceCol);
  const float oldError = previousMapping->error;
  float newError = Error::calculate(source, target, Coordinates(sourceRow, sourceCol),
                                     Coordinates(targetRow, targetCol), patchSize, weights);

  // Adds the omega component to the error.
  const float OMEGA_WEIGHT = 0.001f;
  const float multiplier = OMEGA_WEIGHT / (patchSize * patchSize);
  const float omegaComponentOld = multiplier * Omega::get(omegas, previousMapping->row, previousMapping->col, patchSize);
  const float omegaComponentNew = multiplier * Omega::get(omegas, targetRow, targetCol, patchSize);

  if (newError + omegaComponentNew < oldError + omegaComponentOld) {
    NNFEntry *nextMapping = nextNNF.at(sourceRow, sourceCol);

    // Updates omega for the old mapping.
    Omega::update(omegas, previousMapping->row, previousMapping->col, -1.f, patchSize);

    // If the new error would be lower, updates nextNNF.
    nextMapping->row = targetRow;
    nextMapping->col = targetCol;
    nextMapping->error = newError;

    // Updates omega for the new mapping.
    Omega::update(omegas, nextMapping->row, nextMapping->col, 1.f, patchSize);
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
 * @param optionalBlacklist An optional blacklist (NNF that goes in the other direction). If a
 * location in the blacklist has a valid mapping, it should not be mapped to. A blacklist with zero
 * dimensions is ignored.
 */
template <typename T>
__device__ void tryNeighborOffset(const Image<T> &source, const Image<T> &target,
                                  Image<NNFEntry> &nextNNF, const Image<NNFEntry> &previousNNF,
                                  const int patchSize, const int sourceRow, const int sourceCol,
                                  const int rowOffset, const int colOffset, Image<float> &omegas,
                                  const float *weights) {
  // Gets the neighbor's mapping.
  const int neighborRow = Utilities::clamp(0, sourceRow + rowOffset, source.rows);
  const int neighborCol = Utilities::clamp(0, sourceCol + colOffset, source.cols);
  const NNFEntry &neighborMapping = *previousNNF.constAt(neighborRow, neighborCol);

  // Translates the neighbor's mapping back to get the target coordinates to try.
  const int targetRow = Utilities::clamp(0, neighborMapping.row - rowOffset, target.rows);
  const int targetCol = Utilities::clamp(0, neighborMapping.col - colOffset, target.cols);

  // Tries the mapping.
  tryPatch(source, target, nextNNF, previousNNF, patchSize, sourceRow, sourceCol, targetRow,
           targetCol, omegas, weights);
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
 * @param optionalBlacklist An optional blacklist (NNF that goes in the other direction). If a
 * location in the blacklist has a valid mapping, it should not be mapped to. A blacklist with zero
 * dimensions is ignored.
 */
template <typename T>
__global__ void propagationPassKernel(const Image<T> source, const Image<T> target,
                                      Image<NNFEntry> nextNNF, const Image<NNFEntry> previousNNF,
                                      const int patchSize, const int offset, Image<float> omegas,
                                      const Vec<float> weights) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < source.rows && col < source.cols) {
    // Tries offsetting up, down, left and right.
    tryNeighborOffset(source, target, nextNNF, previousNNF, patchSize, row, col, offset, 0,
        omegas, weights.deviceData);
    tryNeighborOffset(source, target, nextNNF, previousNNF, patchSize, row, col, -offset, 0,
                      omegas, weights.deviceData);
    tryNeighborOffset(source, target, nextNNF, previousNNF, patchSize, row, col, 0, offset,
                      omegas, weights.deviceData);
    tryNeighborOffset(source, target, nextNNF, previousNNF, patchSize, row, col, 0, -offset,
                      omegas, weights.deviceData);
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
 * @param optionalBlacklist An optional blacklist (NNF that goes in the other direction). If a
 * location in the blacklist has a valid mapping, it should not be mapped to. A blacklist with zero
 * dimensions is ignored.
 */
template <typename T>
__global__ void
randomSearchPassKernel(const Image<T> source, const Image<T> target, Image<NNFEntry> nextNNF,
                       const Image<NNFEntry> previousNNF, const int patchSize, const int radius,
                       Image<PCGState> random, Image<float> omegas, const Vec<float> weights) {
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
    tryPatch(source, target, nextNNF, previousNNF, patchSize, row, col, newTargetRow, newTargetCol,
             omegas, weights.deviceData);
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
void run(Image<NNFEntry> &nnf, Image<float> &omegas, const Image<T> &from, const Image<T> &to,
         const Image<PCGState> &random, const int patchSize, const int numIterations,
         const Vec<float> &weights) {
  printf("EBSynthCUDA: Running EBSynthPatchMatch with %d iterations and patch size %d.\n",
         numIterations, patchSize);
  const float initialError = totalError(nnf);

  // Calculates the block size.
  const int BLOCK_SIZE_2D = 16;
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 numBlocks(Utilities::divideRoundUp(from.rows, threadsPerBlock.x),
                       Utilities::divideRoundUp(from.cols, threadsPerBlock.y));

  // Creates a temporary NNF that's used for alternating read/write.
  Image<NNFEntry> tempNNF(nnf.rows, nnf.cols, 1);
  tempNNF.allocate();

  // Runs iterations of EBSynthPatchMatch.
  Image<NNFEntry> *previousNNF = &nnf;
  Image<NNFEntry> *nextNNF = &tempNNF;
  for (int iteration = 0; iteration < numIterations; iteration++) {
    // First, runs three propagation passes with offsets of 4, 2 and 1 respectively.
    propagationPassKernel<T><<<numBlocks, threadsPerBlock>>>(from, to, *nextNNF, *previousNNF,
                                                             patchSize, 4, omegas, weights);
    check(cudaDeviceSynchronize());
    swapNNFs(&previousNNF, &nextNNF);

    propagationPassKernel<T><<<numBlocks, threadsPerBlock>>>(from, to, *nextNNF, *previousNNF,
                                                             patchSize, 2, omegas, weights);
    check(cudaDeviceSynchronize());
    swapNNFs(&previousNNF, &nextNNF);

    propagationPassKernel<T><<<numBlocks, threadsPerBlock>>>(from, to, *nextNNF, *previousNNF,
                                                             patchSize, 1, omegas, weights);
    check(cudaDeviceSynchronize());
    swapNNFs(&previousNNF, &nextNNF);

    // Next, runs a number of random search passes.
    for (int radius = std::max(from.rows, from.cols) / 2; radius > 1; radius /= 2) {
      randomSearchPassKernel<T><<<numBlocks, threadsPerBlock>>>(
          from, to, *nextNNF, *previousNNF, patchSize, radius, random, omegas, weights);
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
  printf("StyLitCUDA: EBSynthPatchMatch reduced error from %f to %f.\n", initialError,
         totalError(nnf));

  // Frees the temporary NNF.
  tempNNF.free();
}

template void run(Image<NNFEntry> &nnf, Image<float> &omegas, const Image<int> &from,
                  const Image<int> &to, const Image<PCGState> &random, const int patchSize,
                  const int numIterations, const Vec<float> &weights);

template void run(Image<NNFEntry> &nnf, Image<float> &omegas, const Image<float> &from,
                  const Image<float> &to, const Image<PCGState> &random, const int patchSize,
                  const int numIterations, const Vec<float> &weights);

} /* namespace EBSynthPatchMatch */
} /* namespace StyLitCUDA */
