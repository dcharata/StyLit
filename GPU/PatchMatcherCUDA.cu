#include "InputPatchMatcherCUDA.h"

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
  inline T *at(int row, int col) {
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
          combinedVector[i] = T(0); // guideVector[i];
        }

        // Adds the style vector to the device image.
        const T *styleVector = &style[numStyleChannels * index];
        for (int i = 0; i < numStyleChannels; i++) {
          combinedVector[i + numGuideChannels] = T(0); // styleVector[i];
        }
      }
    }

    // Copies the reformatted image to the device.
    const int sourcePitch = numTotalChannels * cols * sizeof(T);
    check(cudaMemcpy2D(deviceData, pitch, hostImage, sourcePitch, sourcePitch, rows,
                       cudaMemcpyHostToDevice));

    // Frees the temporary reformatted image on the host.
    check(cudaFreeHost(hostImage));
  }
};

/**
 * @brief The NNFEntryCUDA struct Each NNF entry contains the row that's mapped to, the column
 * that's mapped to, and that mapping's error. NNFs are just ImageCUDAs of NNFEntryCUDAs with one
 * channel.
 */
struct NNFEntryCUDA {
  int row = -1;
  int col = -1;
  float error = std::numeric_limits<float>::max();
};
using NNFCUDA = ImageCUDA<NNFEntryCUDA>;

// This is used to maintain a state for a pseudorandom number generator for each source pixel.
using RandomStateCUDA = ImageCUDA<PCGState>;

/**
 * @brief initializeRandomKernel This kernel is used to initialize the pseudorandom number
 * generator's state.
 * @param random A RandomStateCUDA whose on-device "image" of PCGStates is modified to initialize
 * the NNF.
 */
__global__ void initializeRandomKernel(RandomStateCUDA random) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
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
                                const int targetCol, const int patchSize) {}

/**
 * @brief randomizeNNFKernel Randomly initializes the supplied NNF and calculates the
 * corresponding error values.
 * @param source the source image
 * @param target the target image
 * @param nnf The NNF to modify. Only the NNF's on-device ImageCUDA is modified.
 */
template <typename T>
__global__ void randomizeNNFKernel(const ImageCUDA<T> source, const ImageCUDA<T> target,
                                   const NNFCUDA nnf, const RandomStateCUDA random) {
  printf("You launched an error pass kernel.\n");
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
 * @brief patchMatchCUDA This is what's called from PatchMatcherCUDA. It improves the host NNF.
 * @param input a struct that contains everything patchMatchCUDA needs to run
 */
template <typename T> void patchMatchCUDA(InputPatchMatcherCUDA<T> &input) {
  const int BLOCK_SIZE = 256;

  // Allocates on-device source and target images.
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
  const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 numBlocks(divideRoundUp(input.sourceRows, BLOCK_SIZE),
                       divideRoundUp(input.sourceCols, BLOCK_SIZE));

  // Initializes the pseudorandom number generator.
  RandomStateCUDA random(input.sourceRows, input.sourceCols, 1);
  random.allocate();
  const int sourceArea = input.sourceRows * input.sourceCols;
  initializeRandomKernel<<<numBlocks, threadsPerBlock>>>(random);
  check(cudaDeviceSynchronize());

  // Randomly initializes the even NNF.
  randomizeNNFKernel<<<1, 1>>>(source, target, evenNNF, random);
  check(cudaDeviceSynchronize());

  // Frees on-device memory.
  source.free();
  target.free();
  evenNNF.free();
  oddNNF.free();
  random.free();
}

template void patchMatchCUDA<int>(InputPatchMatcherCUDA<int> &);
template void patchMatchCUDA<float>(InputPatchMatcherCUDA<float> &);
