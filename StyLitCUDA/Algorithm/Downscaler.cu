#include "Downscaler.cuh"

#include "../Utilities/Image.cuh"
#include "../Utilities/Utilities.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {
namespace Downscaler {

template <typename T>
__device__ void sampleBilinear(T *result, const Image<T> &from, const float row, const float col) {
  const int rowFloor = Utilities::restrict(int(row), from.rows);
  const int colFloor = Utilities::restrict(int(col), from.cols);
  const int rowCeil = Utilities::restrict(rowFloor + 1, from.rows);
  const int colCeil = Utilities::restrict(colFloor + 1, from.cols);

  const float rowRemainderForCeil = row - rowFloor;
  const float rowRemainderForFloor = 1.f - rowRemainderForCeil;
  const float colRemainderForCeil = col - colFloor;
  const float colRemainderForFloor = 1.f - colRemainderForCeil;

  const float topLeftWeight = rowRemainderForFloor * colRemainderForFloor;
  const float topRightWeight = rowRemainderForFloor * colRemainderForCeil;
  const float bottomLeftWeight = rowRemainderForCeil * colRemainderForFloor;
  const float bottomRightWeight = rowRemainderForCeil * colRemainderForCeil;

  const T *topLeft = from.constAt(rowFloor, colFloor);
  const T *topRight = from.constAt(rowFloor, colCeil);
  const T *bottomLeft = from.constAt(rowCeil, colFloor);
  const T *bottomRight = from.constAt(rowCeil, colCeil);

  for (int i = 0; i < from.numChannels; i++) {
    result[i] = (T)(topLeft[i] * topLeftWeight + topRight[i] * topRightWeight +
                    bottomLeft[i] * bottomLeftWeight + bottomRight[i] * bottomRightWeight);
  }
}

template <typename T>
__global__ void downscaleKernel(const Image<T> from, Image<T> to, const float rowScale,
                                const float colScale) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < to.rows && col < to.cols) {
    sampleBilinear<T>(to.at(row, col), from, row * rowScale + 0.5f, col * colScale + 0.5f);
  }
}

template <typename T> void downscale(const Image<T> &from, Image<T> to) {
  printf("StyLitCUDA: Downscaling [%d, %d] to [%d, %d].\n", from.rows, from.cols, to.rows, to.cols);

  // Calculates the block size.
  const int BLOCK_SIZE_2D = 16;
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 numBlocks(Utilities::divideRoundUp(to.rows, threadsPerBlock.x),
                       Utilities::divideRoundUp(to.cols, threadsPerBlock.y));

  // Runs the downscaling kernel.
  const float rowScale = float(from.rows) / float(to.rows);
  const float colScale = float(from.cols) / float(to.cols);
  downscaleKernel<T><<<numBlocks, threadsPerBlock>>>(from, to, rowScale, colScale);
  check(cudaDeviceSynchronize());
}

template void downscale(const Image<int> &from, Image<int> to);
template void downscale(const Image<float> &from, Image<float> to);

} /* namespace Downscaler */
} /* namespace StyLitCUDA */
