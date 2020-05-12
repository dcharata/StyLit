#include "Coordinator.cuh"

#include "../Utilities/Image.cuh"
#include "../Utilities/Utilities.cuh"
#include "Applicator.cuh"
#include "Downscaler.cuh"
#include "NNF.cuh"
#include "PatchMatch.cuh"
#include "RandomInitializer.cuh"
#include "ReverseToForwardNNF.cuh"

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
  const int NUM_PATCH_MATCH_ITERATIONS = 6;

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
  NNF::invalidate(forward.levels[coarsestLevel]);
  NNF::randomize<T>(reverse.levels[coarsestLevel], random, a.levels[coarsestLevel],
                    b.levels[coarsestLevel], input.patchSize);
  Applicator::apply<T>(forward.levels[coarsestLevel], b.levels[coarsestLevel],
                       a.levels[coarsestLevel], input.b.numChannels,
                       input.b.numChannels + input.bPrime.numChannels, input.patchSize);

  // Runs StyLit across the pyramid, starting with the lowest level.
  for (int level = coarsestLevel; level >= 0; level--) {
    Image<NNFEntry> &curReverse = reverse.levels[level];
    Image<NNFEntry> &curForward = forward.levels[level];
    Image<T> &curA = a.levels[level];
    Image<T> &curB = b.levels[level];

    // At this stage, the images (A, B, A', B') should be populated.
    // The reverse NNF should be randomized or prepopulated, depending on the pyramid level.
    // The forward NNF should be entirely empty (invalid).
    const int GIVING_UP_THRESHOLD = 10;
    const float STOPPING_THRESHOLD = 0.9f;
    int pixelsMapped = 0;
    for (int i = 0; i < GIVING_UP_THRESHOLD; i++) {
      // Improves the NNF.
      PatchMatch::run(curReverse, &curForward, curA, curB, random, input.patchSize, NUM_PATCH_MATCH_ITERATIONS);
      pixelsMapped += ReverseToForwardNNF::transfer(curReverse, curForward);
      const float fractionFilled = (float)pixelsMapped / (curForward.rows * curForward.cols);
      if (fractionFilled > STOPPING_THRESHOLD) {
        printf("StyLitCUDA: Stopped generating reverse NNFs after %d iterations because forward "
               "NNF is %f percent full (threshold: %f percent).\n",
               i + 1, fractionFilled * 100.f, STOPPING_THRESHOLD * 100.f);
        break;
      }
    }

    // Generates a forward NNF and uses it to fill the remaining entries in the NNF.
    Image<NNFEntry> tempForward(curForward.rows, curForward.cols, 1);
    tempForward.allocate();
    if (level) {
      NNF::upscale(tempForward, forward.levels[level - 1], input.patchSize);
    } else {
      NNF::randomize(tempForward, random, curB, curA, input.patchSize);
    }
    PatchMatch::run(tempForward, nullptr, curB, curA, random, input.patchSize, NUM_PATCH_MATCH_ITERATIONS);
    ReverseToForwardNNF::fill(tempForward, curForward);
    tempForward.free();

    // Upscales or applies the improved NNF, depending on the pyramid level.
    if (level) {
      NNF::upscale(curReverse, reverse.levels[level - 1], input.patchSize);
      Applicator::apply<T>(forward.levels[level - 1], b.levels[level - 1], a.levels[level - 1],
                           input.b.numChannels, input.b.numChannels + input.bPrime.numChannels,
                           input.patchSize);
    } else {
      // For the finest pyramid level, the NNF is applied to produce the final B'.
      Applicator::apply<T>(curForward, curB, curA, input.b.numChannels,
                           input.b.numChannels + input.bPrime.numChannels, input.patchSize);
    }
  }

  // Copies B' back to the caller.
  std::vector<InterfaceImage<T>> bImagesPrime(1);
  bImagesPrime[0] = input.bPrime;
  b.levels[0].retrieveChannels(bImagesPrime, input.b.numChannels);
}

template <typename T> Coordinator<T>::~Coordinator() { random.free(); }

template class Coordinator<int>;
template class Coordinator<float>;

void runCoordinator_float(InterfaceInput<float> &input) { Coordinator<float> coordinator(input); }

} /* namespace StyLitCUDA */
