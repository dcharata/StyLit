#include "Error.cuh"

#include "../Utilities/Utilities.cuh"

#include <cuda_runtime.h>

namespace StyLitCUDA {
namespace Error {

template <typename T>
__device__ float calculate(const Image<T> &x, const Image<T> &y, Coordinates inX, Coordinates inY, const int patchSize) {
  // Clamps the patch coordinates so that every pixel inside a patch is in bounds.
  const int halfPatch = patchSize / 2;
  const int xRowLimit = x.rows - halfPatch;
  const int xColLimit = x.cols - halfPatch;
  const int yRowLimit = y.rows - halfPatch;
  const int yColLimit = y.cols - halfPatch;

  // Sums up the error across the patch.
  float error = 0.f;
  for (int rowOffset = -halfPatch; rowOffset <= halfPatch; rowOffset++) {
    for (int colOffset = -halfPatch; colOffset <= halfPatch; colOffset++) {
      // Calculates the coordinates in the patch.
      const int xPatchRow = Utilities::clamp(halfPatch, inX.row - rowOffset, xRowLimit);
      const int xPatchCol = Utilities::clamp(halfPatch, inX.col - colOffset, xColLimit);
      const int yPatchRow = Utilities::clamp(halfPatch, inY.row - rowOffset, yRowLimit);
      const int yPatchCol = Utilities::clamp(halfPatch, inY.col - colOffset, yColLimit);

      // Adds the errors for each channel.
      const T *xVector = x.constAt(xPatchRow, xPatchCol);
      const T *yVector = y.constAt(yPatchRow, yPatchCol);
      for (int channel = 0; channel < x.numChannels; channel++) {
        const float difference = xVector[channel] - yVector[channel];
        error += difference * difference;
      }
    }
  }
  return error;
}

template __device__ float calculate(const Image<int> &x, const Image<int> &y, Coordinates inX, Coordinates inY, const int patchSize);
template __device__ float calculate(const Image<float> &x, const Image<float> &y, Coordinates inX, Coordinates inY, const int patchSize);

} /* namespace Error */
} /* namespace StyLitCUDA */
