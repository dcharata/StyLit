#include "Applicator.cuh"

#include "../Utilities/Utilities.cuh"

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {
namespace Applicator {

template <typename T>
__global__ void applyKernel(const Image<NNFEntry> nnf, Image<T> from, const Image<T> to,
                            const int startChannel, const int endChannel, const int patchSize) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < to.rows && col < to.cols) {
    // Counterintuitively, from is the image that's being populated.
    // Zeroes out the pixel that will be populated.
    T *fromVector = from.at(row, col);
    for (int channel = startChannel; channel < endChannel; channel++) {
      fromVector[channel] = (T)0;
    }

    // Gets the mapping for each pixel within the patch.
    int denominator = 0.f;
    const int halfPatch = patchSize / 2;
    const int fromRowStart = Utilities::clamp(0, row - halfPatch, from.rows);
    const int fromRowEnd = Utilities::clamp(0, row + halfPatch, from.rows);
    const int fromColStart = Utilities::clamp(0, col - halfPatch, from.cols);
    const int fromColEnd = Utilities::clamp(0, col + halfPatch, from.cols);
    for (int fromRow = fromRowStart; fromRow < fromRowEnd; fromRow++) {
      for (int fromCol = fromColStart; fromCol < fromColEnd; fromCol++) {
        // Offsets the mapping so it corresponds to the pixel that will be populated.
        const NNFEntry *mapping = nnf.constAt(fromRow, fromCol);
        const int toRow = mapping->row + row - fromRow;
        const int toCol = mapping->col + col - fromCol;

        // Includes the pixel if it's within range.
        if (toRow > 0 && toRow < to.rows && toCol > 0 && toCol < to.cols) {
          const T *toVector = to.constAt(toRow, toCol);
          for (int channel = startChannel; channel < endChannel; channel++) {
            fromVector[channel] += toVector[channel];
          }
          denominator += 1.f;
        }
      }
    }

    // Divides by the number of pixels sampled.
    for (int channel = startChannel; channel < endChannel; channel++) {
      fromVector[channel] = (T)(fromVector[channel] / denominator);
    }
  }
}

template <typename T>
void apply(const Image<NNFEntry> &nnf, Image<T> &from, const Image<T> &to, const int startChannel,
           const int endChannel, const int patchSize) {
  assert(from.rows == nnf.rows && from.cols == nnf.cols);
  assert(startChannel >= 0 && startChannel < endChannel && endChannel <= from.numChannels);
  assert(from.numChannels == to.numChannels);
  printf("StyLitCUDA: Applying image with dimensions [%d, %d] to image with dimensions [%d, %d] "
         "using patch size %d.\n",
         from.rows, from.cols, to.rows, to.cols, patchSize);

  // Calculates the block size.
  const int BLOCK_SIZE_2D = 16;
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 numBlocks(Utilities::divideRoundUp(to.rows, threadsPerBlock.x),
                       Utilities::divideRoundUp(to.cols, threadsPerBlock.y));

  // Runs the applicator kernel.
  applyKernel<T>
      <<<numBlocks, threadsPerBlock>>>(nnf, from, to, startChannel, endChannel, patchSize);
  check(cudaDeviceSynchronize());
}

template void apply(const Image<NNFEntry> &nnf, Image<int> &from, const Image<int> &to,
                    const int startChannel, const int endChannel, const int patchSize);
template void apply(const Image<NNFEntry> &nnf, Image<float> &from, const Image<float> &to,
                    const int startChannel, const int endChannel, const int patchSize);

} /* namespace Applicator */
} /* namespace StyLitCUDA */
