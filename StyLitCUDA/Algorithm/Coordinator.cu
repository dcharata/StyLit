#include "Coordinator.cuh"

#include "../Utilities/Image.cuh"
#include "../Utilities/Utilities.cuh"
#include "Applicator.cuh"
#include "Downscaler.cuh"
#include "NNF.cuh"
#include "PatchMatch.cuh"
#include "RandomInitializer.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

namespace StyLitCUDA {

template <typename T>
Coordinator<T>::Coordinator(InterfaceInput<T> &input)
    : input(input), a(input.a.rows, input.a.cols, input.a.numChannels + input.aPrime.numChannels,
                      input.numLevels),
      b(input.b.rows, input.b.cols, input.b.numChannels + input.bPrime.numChannels,
        input.numLevels),
      random(std::max(input.a.rows, input.b.rows), std::max(input.a.cols, input.b.cols), 1),
      forward(input.b.rows, input.b.cols, 1, input.numLevels),
      reverse(input.a.rows, input.a.cols, 1, input.numLevels) {
  // Loads the images into A and B.
  // A contains both A and A'.
  // B contains only B (since B' is filled in by StyLit).
  std::vector<InterfaceImage<T>> aImages(2);
  aImages[0] = input.a;
  aImages[1] = input.aPrime;
  a.levels[0].populateChannels(aImages, 0);
  std::vector<InterfaceImage<T>> bImages(1);
  bImages[0] = input.b;
  b.levels[0].populateChannels(bImages, 0);

  // Downscales the images to form the pyramid.
  for (int level = 0; level < a.levels.size() - 1; level++) {
    downscale(a.levels[level], a.levels[level + 1]);
  }
  for (int level = 0; level < b.levels.size() - 1; level++) {
    downscale(b.levels[level], b.levels[level + 1]);
  }

  // Initializes the PCG state for pseudorandom number generation.
  // Since random is passed into kernels directly, it can't use RAII.
  random.allocate();
  initializeRandomState(random);

  // Randomizes the NNFs and populates B' at the coarsest pyramid level.
  const int coarsestLevel = input.numLevels - 1;
  NNF::randomize<T>(forward.levels[coarsestLevel], random, b.levels[coarsestLevel],
                    a.levels[coarsestLevel], input.patchSize);
  NNF::randomize<T>(reverse.levels[coarsestLevel], random, a.levels[coarsestLevel],
                    b.levels[coarsestLevel], input.patchSize);
  Applicator::apply<T>(forward.levels[coarsestLevel], b.levels[coarsestLevel],
                       a.levels[coarsestLevel], input.b.numChannels,
                       input.b.numChannels + input.bPrime.numChannels, input.patchSize);

  // Runs StyLit across the pyramid, starting with the lowest level.
  for (int level = coarsestLevel; level >= 0; level--) {
    // At this stage, all NNFs and images (A, B, A', B') should be populated.
    // Improves the NNF.
    PatchMatch::run(forward.levels[level], nullptr, b.levels[level], a.levels[level], random,
                    input.patchSize, 6);

    // Upscales or applies the improved NNF, depending on the pyramid level.
    if (level) {
      NNF::upscale(forward.levels[level], forward.levels[level - 1]);
      Applicator::apply<T>(forward.levels[level - 1], b.levels[level - 1], a.levels[level - 1],
                           input.b.numChannels, input.b.numChannels + input.bPrime.numChannels,
                           input.patchSize);
    } else {
      // For the finest pyramid level, the NNF is applied to produce the final B'.
      Applicator::apply<T>(forward.levels[0], b.levels[0], a.levels[0], input.b.numChannels,
                           input.b.numChannels + input.bPrime.numChannels, input.patchSize);
    }
  }

  // Copies B' back to the caller.
  // TODO: (this is currently configured to export the lowest resolution)
  std::vector<InterfaceImage<T>> bImagesPrime(1);
  bImagesPrime[0] = input.bPrime;
  b.levels[1].retrieveChannels(bImagesPrime, input.b.numChannels);
}

template <typename T> Coordinator<T>::~Coordinator() { random.free(); }

template class Coordinator<int>;
template class Coordinator<float>;

void runCoordinator_float(InterfaceInput<float> &input) { Coordinator<float> coordinator(input); }

} /* namespace StyLitCUDA */
