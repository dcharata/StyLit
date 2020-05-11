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
    // Gets the NNF mapping.
    const NNFEntry *entry = nnf.constAt(row, col);
    const int halfPatch = patchSize / 2;
    const int toRowStart = Utilities::clamp(0, entry->row - halfPatch, to.rows);
    const int toRowEnd = Utilities::clamp(0, entry->row + halfPatch, to.rows);
    const int toColStart = Utilities::clamp(0, entry->col - halfPatch, to.cols);
    const int toColEnd = Utilities::clamp(0, entry->col + halfPatch, to.cols);

    // Zeroes out the pixel that will be populated.
    T *fromVector = from.at(row, col);
    for (int channel = startChannel; channel < endChannel; channel++) {
      fromVector[channel] = (T)0;
    }

    // Adds the values from the sampled region of to.
    for (int toRow = toRowStart; toRow < toRowEnd; toRow++) {
      for (int toCol = toColStart; toCol < toColEnd; toCol++) {
        const T *toVector = to.constAt(toRow, toCol);
        for (int channel = startChannel; channel < endChannel; channel++) {
          fromVector[channel] += toVector[channel];
        }
      }
    }

    // Divides by the number of pixels sampled.
    const float area = (float)((toRowEnd - toRowStart) * (toColEnd - toColStart));
    for (int channel = startChannel; channel < endChannel; channel++) {
      fromVector[channel] = (T)(fromVector[channel] / area);
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
