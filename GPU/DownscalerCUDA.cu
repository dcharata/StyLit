#include <stdio.h>

/**
 * @brief restrict Restricts the value to [0, max).
 * @param value the image to sample
 * @param max the row to sample at
 * @return [0, max)
 */
__device__ int restrict(int value, int max) {
  return (value >= max) ? max - 1 : value;
}

template<typename T>
__device__ T interpolate(float aWeight,
                         T a, float bWeight,
                         T b, float cWeight,
                         T c, float dWeight,
                         T d) {
  // This will in fact work for both integers and floats.
  return (T)(a * aWeight + b * bWeight + c * cWeight + d * dWeight);
}

template<typename T>
__device__ void sampleBilinear(const T *full, float row, float col, T *result, int fullRows, int fullCols, int numChannels) {
  const int rowFloor = restrict(int(row), fullRows);
  const int colFloor = restrict(int(col), fullCols);
  const int rowCeil = restrict(rowFloor + 1, fullRows);
  const int colCeil = restrict(colFloor + 1, fullCols);
  const float rowRemainderForCeil = row - rowFloor;
  const float rowRemainderForFloor = 1.f - rowRemainderForCeil;
  const float colRemainderForCeil = col - colFloor;
  const float colRemainderForFloor = 1.f - colRemainderForCeil;
  for (int i = 0; i < numChannels; i++) {
    const float topLeftWeight = rowRemainderForFloor * colRemainderForFloor;
    const T topLeft = full[numChannels * (rowFloor * fullCols + colFloor) + i];

    const float topRightWeight = rowRemainderForFloor * colRemainderForCeil;
    const T topRight = full[numChannels * (rowFloor * fullCols + colCeil) + i];

    const float bottomLeftWeight = rowRemainderForCeil * colRemainderForFloor;
    const T bottomLeft = full[numChannels * (rowCeil * fullCols + colFloor) + i];

    const float bottomRightWeight = rowRemainderForCeil * colRemainderForCeil;
    const T bottomRight = full[numChannels * (rowCeil * fullCols + colCeil) + i];

    result[i] = interpolate(topLeftWeight, topLeft, topRightWeight, topRight,
                            bottomLeftWeight, bottomLeft, bottomRightWeight, bottomRight);
  }
}

template<typename T>
__global__ void downscalerKernel(const T *full, T *half, int numChannels, int fullRows, int fullCols, int halfRows, int halfCols, float rowScale, float colScale){
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = index / halfCols;
  const int col = index % halfCols;
  const int halfStart = numChannels * (row * halfCols + col);
  sampleBilinear<T>(full, row * rowScale + 0.5f, col * colScale + 0.5f, &half[halfStart], fullRows, fullCols, numChannels);
}

template<typename T>
int downscaleCUDA(const T *full, T *half, int numChannels, int fullRows, int fullCols, int halfRows, int halfCols) {
  // Allocates shared memory for the full and half images.
  const int fullSize = numChannels * fullRows * fullCols;
  const int numPixelsInHalf = halfRows * halfCols;
  const int halfSize = numChannels * numPixelsInHalf;
  const int fullSizeInBytes = fullSize * sizeof(T);
  const int halfSizeInBytes = halfSize * sizeof(T);
  const T *fullManaged;
  T *halfManaged;
  cudaMallocManaged(&fullManaged, fullSize * sizeof(T *));
  cudaMallocManaged(&halfManaged, halfSize * sizeof(T *));

  // Copies the images to shared memory.
  memcpy((void *)fullManaged, (void *)full, fullSizeInBytes);
  memcpy((void *)halfManaged, (void *)half, halfSizeInBytes);

  const int BLOCK_SIZE = 256;
  const int numBlocks = (numPixelsInHalf + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const float colScale = float(fullCols) / float(halfCols);
  const float rowScale = float(fullRows) / float(halfRows);
  downscalerKernel<T><<<numBlocks, 256>>>(fullManaged, halfManaged, numChannels, fullRows, fullCols, halfRows, halfCols, rowScale, colScale);
  cudaDeviceSynchronize();

  // Copies the images back to host memory.
  memcpy((void *)full, (void *)fullManaged, fullSizeInBytes);
  memcpy((void *)half, (void *)halfManaged, halfSizeInBytes);

  // Frees the shared memory.
  cudaFree((void *)fullManaged);
  cudaFree((void *)halfManaged);
  return 0;
}

// The templated versions have to be declared explicitly.
template int downscaleCUDA<int>(const int *, int *, int, int, int, int, int);
template int downscaleCUDA<float>(const float *, float *, int, int, int, int, int);
