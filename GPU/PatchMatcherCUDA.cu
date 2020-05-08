#include "InputPatchMatcherCUDA.h"

#include <algorithm>
#include <limits>
#include <stdint.h>
#include <stdio.h>

// Together with the macro, assertCUDA prints CUDA errors that occur to stderr.
#define check(ans)                                                                                 \
  { assertCUDA((ans), __FILE__, __LINE__); }
inline void assertCUDA(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "epic CUDA fail: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

/**
 * @brief The PCGState struct This holds the state for PCG pseudorandom number generation. This is
 * copied from EBSynth.
 */
struct PCGState {
  uint64_t state;
  uint64_t increment;
};

__device__ void pcgAdvance(PCGState *rng) {
  rng->state = rng->state * 6364136223846793005ULL + rng->increment;
}

__device__ uint32_t pcgOutput(uint64_t state) {
  return (uint32_t)(((state >> 22u) ^ state) >> ((state >> 61u) + 22u));
}

__device__ uint32_t pcgRand(PCGState *rng) {
  uint64_t oldstate = rng->state;
  pcgAdvance(rng);
  return pcgOutput(oldstate);
}

__device__ void pcgInit(PCGState *rng, uint64_t seed, uint64_t stream) {
  rng->state = 0U;
  rng->increment = (stream << 1u) | 1u;
  pcgAdvance(rng);
  rng->state += seed;
  pcgAdvance(rng);
}

/**
 * @brief The ImageCUDA struct This is how an image is represented within PatchMatcherCUDA. It
 * exists mostly to manage copying memory between the host and the device and make indexing into
 * device images easier.
 */
template <typename T> struct ImageCUDA {
  // Allocates the device memory.
  ImageCUDA(int rows, int cols, int numChannels)
      : rows(rows), cols(cols), numChannels(numChannels) {}

  /**
   * @brief allocate Allocates the on-device memory associated with this ImageCUDA. This isn't in
   * ImageCUDA's constructor because it should only be triggered intentionally by the host, not when
   * an ImageCUDA is passed to the host via a kernel.
   */
  void allocate() {
    check(cudaMallocPitch(&deviceData, &pitch, numChannels * cols * sizeof(T), rows));
  }

  /**
   * @brief free Frees the on-device memory associated with this ImageCUDA. This isn't in
   * ImageCUDA's destructor because it should only be triggered intentionally by the host, not when
   * an ImageCUDA is passed to the device via a kernel.
   */
  void free() { check(cudaFree(deviceData)); }

  // Returns a pointer to the feature vector at the given coordinates.
  __device__ inline T *at(int row, int col) {
    // This is adapted from NVidia's website.
    T *rowStart = (T *)((char *)deviceData + row * pitch);
    return &rowStart[col * numChannels];
  }

  // Returns a pointer to the feature vector at the given coordinates, but with a bit of const
  // sprinkled in so that the compiler is happy.
  __device__ inline const T *constAt(int row, int col) const {
    // This is adapted from NVidia's website.
    T *rowStart = (T *)((char *)deviceData + row * pitch);
    return &rowStart[col * numChannels];
  }

  // The image's data, but padded and copied to the device.
  void *deviceData;

  // This is the pitch that cudaMallocPitch returns.
  size_t pitch;

  // The number of rows in the image, i.e. its height.
  const int rows;

  // The number of columns in the image, i.e. its width.
  const int cols;

  // The number of channels in the image.
  const int numChannels;

  /**
   * @brief populateWithStyleAndSource Populates an ImageCUDA with the corresponding guide and style
   * images. Whereas the guide and style images are arrays of feature vectors with numGuideChannels
   * and numStyleChannels entries respectively, image becomes an array of feature vectors with
   * (numGuideChannels + numStyleChannels) entries. This makes the PatchMatch algorithm below
   * simpler and could theoretically improve memory locality slightly.
   * @param guide the guide image
   * @param style the style image
   * @param rows the guide/style images' number of rows
   * @param cols the guide/style images' number of columns
   * @param numGuideChannels the number of channels in the guide image
   * @param numStyleChannels the number of channels in the style iamge
   */
  void populateWithStyleAndSource(const T *guide, const T *style, const int rows, const int cols,
                                  const int numGuideChannels, const int numStyleChannels) {
    // Temporarily allocates space for the reformatted image on the host.
    const int numTotalChannels = numGuideChannels + numStyleChannels;
    T *hostImage;
    check(cudaMallocHost(&hostImage, rows * cols * numTotalChannels * sizeof(T)));

    // Reformats the image on the host.
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        const int index = row * cols + col;
        T *combinedVector = &hostImage[numTotalChannels * index];

        // Adds the guide vector to the device image.
        const T *guideVector = &guide[numGuideChannels * index];
        for (int i = 0; i < numGuideChannels; i++) {
          combinedVector[i] = guideVector[i];
        }

        // Adds the style vector to the device image.
        const T *styleVector = &style[numStyleChannels * index];
        for (int i = 0; i < numStyleChannels; i++) {
          combinedVector[i + numGuideChannels] = styleVector[i];
        }
      }
    }

    // Copies the reformatted image to the device.
    const int hostPitch = numTotalChannels * cols * sizeof(T);
    check(cudaMemcpy2D(deviceData, pitch, hostImage, hostPitch, hostPitch, rows,
                       cudaMemcpyHostToDevice));

    // Frees the temporary reformatted image on the host.
    check(cudaFreeHost(hostImage));
  }
};

/**
 * @brief The NNFMappingCUDA struct Each NNF entry contains the row that's mapped to, the column
 * that's mapped to, and that mapping's error. NNFs are just ImageCUDAs of NNFMappingCUDAs with one
 * channel.
 */
struct NNFMappingCUDA {
  int row = -1;
  int col = -1;
  float error = std::numeric_limits<float>::max();
};
using NNFCUDA = ImageCUDA<NNFMappingCUDA>;

// This is used to maintain a state for a pseudorandom number generator for each source pixel.
using RandomStateCUDA = ImageCUDA<PCGState>;

/**
 * @brief initializeRandomKernel This kernel is used to initialize the pseudorandom number
 * generator's state.
 * @param random A RandomStateCUDA whose on-device "image" of PCGStates is modified to initialize
 * the NNF.
 */
__global__ void initializeRandomKernel(RandomStateCUDA random) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < random.rows && col < random.cols) {
    pcgInit(random.at(row, col), 1361, row * random.cols + col);
  }
}

/**
 * @brief clamp Clamps value to the range [minInclusive, maxExclusive).
 * @param minInclusive the inclusive minimum value
 * @param value the value to clamp
 * @param maxExclusive the exclusive maximum value
 * @return value clamped to [minInclusive, maxExclusive)
 */
__device__ inline int clamp(const int minInclusive, const int value, const int maxExclusive) {
  if (value < minInclusive) {
    return minInclusive;
  }
  if (value >= maxExclusive) {
    return maxExclusive - 1;
  }
  return value;
}

/**
 * @brief calculateError Calculates the error for the specified NNF mapping. A mapping's error is
 * defined as the sum of errors across the pixels in the corresponding source and target patches. A
 * pixel's error is defined as the sum of errors across its channels. A channel's error is defined
 * as the square of the difference between its value in the source and its value in the target. If
 * part of a patch is outside of an image, its coordinates are clamped to be within the image.
 * @param source the source image
 * @param target the target image
 * @param sourceRow the source row
 * @param sourceCol the source column
 * @param targetRow the target row
 * @param targetCol the target column
 * @param patchSize the patch's width/height
 * @return the mapping's error
 */
template <typename T>
__device__ float calculateError(const ImageCUDA<T> &source, const ImageCUDA<T> &target,
                                const int sourceRow, const int sourceCol, const int targetRow,
                                const int targetCol, const int patchSize) {
  // Clamps the patch coordinates so that every pixel inside a patch is in bounds.
  const int halfPatch = patchSize / 2;
  const int sourceRowLimit = source.rows - halfPatch;
  const int sourceColLimit = source.cols - halfPatch;
  const int targetRowLimit = target.rows - halfPatch;
  const int targetColLimit = target.cols - halfPatch;

  // Sums up the error across the patch.
  float error = 0.f;
  for (int rowOffset = -halfPatch; rowOffset <= halfPatch; rowOffset++) {
    for (int colOffset = -halfPatch; colOffset <= halfPatch; colOffset++) {
      // Calculates the coordinates in the patch.
      const int patchSourceRow = clamp(halfPatch, sourceRow - rowOffset, sourceRowLimit);
      const int patchSourceCol = clamp(halfPatch, sourceCol - colOffset, sourceColLimit);
      const int patchTargetRow = clamp(halfPatch, targetRow - rowOffset, targetRowLimit);
      const int patchTargetCol = clamp(halfPatch, targetCol - colOffset, targetColLimit);

      // Adds the errors for each channel.
      const T *sourceVector = source.constAt(patchSourceRow, patchSourceCol);
      const T *targetVector = target.constAt(patchTargetRow, patchTargetCol);
      for (int channel = 0; channel < source.numChannels; channel++) {
        const float difference = sourceVector[channel] - targetVector[channel];
        error += difference * difference;
      }
    }
  }
  return error;
}

/**
 * @brief randomizeNNFKernel Randomly initializes the supplied NNF and calculates the
 * corresponding error values.
 * @param source the source image
 * @param target the target image
 * @param nnf The NNF to modify. Only the NNF's on-device ImageCUDA is modified.
 * @param patchSize the patch's width/height
 */
template <typename T>
__global__ void randomizeNNFKernel(const ImageCUDA<T> source, const ImageCUDA<T> target,
                                   NNFCUDA nnf, RandomStateCUDA random, const int patchSize) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < random.rows && col < random.cols) {
    // Generates a random mapping.
    const int mappedRow = pcgRand(random.at(row, col)) % target.rows;
    const int mappedCol = pcgRand(random.at(row, col)) % target.cols;

    // Fills the random mapping in.
    NNFMappingCUDA *entry = nnf.at(row, col);
    entry->row = mappedRow;
    entry->col = mappedCol;
    entry->error = calculateError(source, target, row, col, mappedRow, mappedCol, patchSize);
  }
}

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
__device__ void tryPatch(const ImageCUDA<T> &source, const ImageCUDA<T> &target, NNFCUDA &nextNNF,
                         const NNFCUDA &previousNNF, const int patchSize, const int sourceRow,
                         const int sourceCol, const int targetRow, const int targetCol) {
  // Calculates the error for the new mapping and compares it with the existing error.
  const float oldError = previousNNF.constAt(sourceRow, sourceCol)->error;
  const float newError =
      calculateError(source, target, sourceRow, sourceCol, targetRow, targetCol, patchSize);
  if (newError <= oldError) {
    // If the new error would be lower, updates nextNNF.
    NNFMappingCUDA *nextMapping = nextNNF.at(sourceRow, sourceCol);
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
__device__ void tryNeighborOffset(const ImageCUDA<T> &source, const ImageCUDA<T> &target,
                                  NNFCUDA &nextNNF, const NNFCUDA &previousNNF, const int patchSize,
                                  const int sourceRow, const int sourceCol, const int rowOffset,
                                  const int colOffset) {
  // Gets the neighbor's mapping.
  const int neighborRow = clamp(0, sourceRow + rowOffset, source.rows);
  const int neighborCol = clamp(0, sourceCol + colOffset, source.cols);
  const NNFMappingCUDA &neighborMapping = *previousNNF.constAt(neighborRow, neighborCol);

  // Translates the neighbor's mapping back to get the target coordinates to try.
  const int targetRow = clamp(0, neighborMapping.row - rowOffset, target.rows);
  const int targetCol = clamp(0, neighborMapping.col - colOffset, target.cols);

  // Tries the mapping.
  tryPatch(source, target, nextNNF, previousNNF, patchSize, sourceRow, sourceCol, targetRow,
           targetCol);
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
__global__ void randomSearchPassKernel(const ImageCUDA<T> source, const ImageCUDA<T> target,
                                       NNFCUDA nextNNF, const NNFCUDA previousNNF,
                                       const int patchSize, const int radius,
                                       RandomStateCUDA random) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < source.rows && col < source.cols) {
    // Gets the current mapping.
    const NNFMappingCUDA *previousMapping = previousNNF.constAt(row, col);

    // Randomly shifts the mapping within the radius.
    PCGState *randomState = random.at(row, col);
    const int range = 2 * radius;
    const int rowShift = pcgRand(randomState) % range - radius;
    const int colShift = pcgRand(randomState) % range - radius;
    const int newTargetRow = clamp(0, previousMapping->row + rowShift, target.rows);
    const int newTargetCol = clamp(0, previousMapping->col + colShift, target.cols);

    // Tries the shifted patch.
    tryPatch(source, target, nextNNF, previousNNF, patchSize, row, col, newTargetRow, newTargetCol);
  }
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
__global__ void propagationPassKernel(const ImageCUDA<T> source, const ImageCUDA<T> target,
                                      NNFCUDA nextNNF, const NNFCUDA previousNNF,
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
 * @brief divideRoundUp Does integer division, but rounds up instead of down. Only works on
 * positive numbers.
 * @param a the dividend
 * @param b the divisor
 * @return a divided by b, except it rounds up
 */
int divideRoundUp(int a, int b) { return (a + b - 1) / b; }

/**
 * @brief swapNNFs Swaps the pointers. This is used to alternate which NNF is read from and which
 * NNF is written to.
 * @param a one of the NNFs
 * @param b the other NNF
 */
void swapNNFs(NNFCUDA **a, NNFCUDA **b) {
  NNFCUDA *temp = *a;
  *a = *b;
  *b = temp;
}

/**
 * @brief patchMatchCUDA This is what's called from PatchMatcherCUDA. It improves the host NNF.
 * @param input a struct that contains everything patchMatchCUDA needs to run
 */
template <typename T> void patchMatchCUDA(InputPatchMatcherCUDA<T> &input) {
  const int BLOCK_SIZE_2D = 16;
  const int PATCH_SIZE = 5;
  const int NUM_ITERATIONS = 6;

  // Allocates on-device source and target images and copies over the on-host images' contents.
  const int numTotalChannels = input.numGuideChannels + input.numStyleChannels;
  ImageCUDA<T> source(input.sourceRows, input.sourceCols, numTotalChannels);
  source.allocate();
  ImageCUDA<T> target(input.targetRows, input.targetCols, numTotalChannels);
  target.allocate();
  source.populateWithStyleAndSource(input.hostGuideSource, input.hostStyleSource, input.sourceRows,
                                    input.sourceCols, input.numGuideChannels,
                                    input.numStyleChannels);
  target.populateWithStyleAndSource(input.hostGuideTarget, input.hostStyleTarget, input.targetRows,
                                    input.targetCols, input.numGuideChannels,
                                    input.numStyleChannels);

  // Allocates the on-device NNFs.
  NNFCUDA evenNNF(input.sourceRows, input.sourceCols, 1);
  evenNNF.allocate();
  NNFCUDA oddNNF(input.sourceRows, input.sourceCols, 1);
  oddNNF.allocate();

  // Initializes kernel dimensions for operations that are per-pixel in the source.
  const dim3 threadsPerBlock(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
  const dim3 numBlocks(divideRoundUp(input.sourceRows, threadsPerBlock.x),
                       divideRoundUp(input.sourceCols, threadsPerBlock.y));

  // Initializes the pseudorandom number generator.
  RandomStateCUDA random(input.sourceRows, input.sourceCols, 1);
  random.allocate();
  const int sourceArea = input.sourceRows * input.sourceCols;
  initializeRandomKernel<<<numBlocks, threadsPerBlock>>>(random);
  check(cudaDeviceSynchronize());

  // Randomly initializes the even NNF.
  // Copies the result over to the odd NNF.
  randomizeNNFKernel<<<numBlocks, threadsPerBlock>>>(source, target, evenNNF, random, PATCH_SIZE);
  check(cudaDeviceSynchronize());
  check(cudaMemcpy2D(oddNNF.deviceData, oddNNF.pitch, evenNNF.deviceData, evenNNF.pitch,
                     oddNNF.cols * sizeof(NNFMappingCUDA), oddNNF.rows, cudaMemcpyDeviceToDevice));

  // Runs iterations of PatchMatch.
  NNFCUDA *previousNNF = &evenNNF;
  NNFCUDA *nextNNF = &oddNNF;
  for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
    // First, runs three propagation passes with offsets of 4, 2 and 1 respectively.
    propagationPassKernel<T>
        <<<numBlocks, threadsPerBlock>>>(source, target, *nextNNF, *previousNNF, PATCH_SIZE, 4);
    check(cudaDeviceSynchronize());
    swapNNFs(&previousNNF, &nextNNF);

    propagationPassKernel<T>
        <<<numBlocks, threadsPerBlock>>>(source, target, *nextNNF, *previousNNF, PATCH_SIZE, 2);
    check(cudaDeviceSynchronize());
    swapNNFs(&previousNNF, &nextNNF);

    propagationPassKernel<T>
        <<<numBlocks, threadsPerBlock>>>(source, target, *nextNNF, *previousNNF, PATCH_SIZE, 1);
    check(cudaDeviceSynchronize());
    swapNNFs(&previousNNF, &nextNNF);

    // Next, runs a number of random search passes.
    for (int radius = std::max(input.sourceRows, input.sourceCols) / 2; radius > 1; radius /= 2) {
      randomSearchPassKernel<T><<<numBlocks, threadsPerBlock>>>(
          source, target, *nextNNF, *previousNNF, PATCH_SIZE, radius, random);
      check(cudaDeviceSynchronize());
      swapNNFs(&previousNNF, &nextNNF);
    }
  }

  // Since every NNF-altering call has a call to swapNNFs after it, previousNNF is actually the
  // most up-to-date NNF. It's swapped back here to make the code below less confusing.
  swapNNFs(&previousNNF, &nextNNF);

  // Copies the resulting NNF back to the host.
  // This assumes that host NNF coordinates are tuples of ints.
  float debugError = 0.f;
  const NNFMappingCUDA *hostReturnNNF;
  check(
      cudaMallocHost(&hostReturnNNF, input.sourceRows * input.sourceCols * sizeof(NNFMappingCUDA)));
  const int hostPitch = input.sourceCols * sizeof(NNFMappingCUDA);
  check(cudaMemcpy2D((void *)hostReturnNNF, hostPitch, nextNNF->deviceData, nextNNF->pitch,
                     hostPitch, input.sourceRows, cudaMemcpyDeviceToHost));
  for (int row = 0; row < input.sourceRows; row++) {
    for (int col = 0; col < input.sourceCols; col++) {
      const int index = row * input.sourceCols + col;
      const NNFMappingCUDA &tempEntry = hostReturnNNF[index];
      int *returnEntry = &input.hostNNF[index * 2];
      returnEntry[0] = tempEntry.row;
      returnEntry[1] = tempEntry.col;
      debugError += tempEntry.error;
    }
  }
  check(cudaFreeHost((void *)hostReturnNNF));
  printf("Total error after PatchMatchCUDA: %f\n", debugError);

  // Frees on-device memory.
  source.free();
  target.free();
  evenNNF.free();
  oddNNF.free();
  random.free();
}

template void patchMatchCUDA<int>(InputPatchMatcherCUDA<int> &);
template void patchMatchCUDA<float>(InputPatchMatcherCUDA<float> &);
