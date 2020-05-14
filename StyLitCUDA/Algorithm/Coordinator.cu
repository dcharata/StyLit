#include "Coordinator.cuh"

#include "../Utilities/Image.cuh"
#include "../Utilities/Utilities.cuh"
#include "../Utilities/Vec.cuh"
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
  // Sets up the weights.
  std::vector<float> hostWeights;
  for (int i = 0; i < input.a.numChannels + input.aPrime.numChannels; i++) {
    if (i < input.a.numChannels) {
      hostWeights.push_back(input.guideWeights[i]);
    } else {
      hostWeights.push_back(input.styleWeights[i - input.a.numChannels]);
    }
  }
  Vec<float> weights(input.a.numChannels + input.aPrime.numChannels);
  weights.deviceAllocate();
  weights.toDevice(hostWeights.data());

  const int NUM_PATCH_MATCH_ITERATIONS = input.patchMatchIterations;
  const int NUM_OPTIMIZATIONS_PER_LEVEL = input.optimizationIterations;

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
  for (int level = 0; level < input.numLevels - 1; level++) {
    downscale(a.levels[level], a.levels[level + 1]);
    downscale(b.levels[level], b.levels[level + 1]);
  }

  // Initializes the PCG state for pseudorandom number generation.
  // Since random is passed into kernels directly, it can't use RAII.
  random.allocate();
  initializeRandomState(random);

  // Randomizes the NNFs and populates B' at the coarsest pyramid level.
  const int coarsestLevel = input.numLevels - 1;

  NNF::randomize<T>(reverse.levels[coarsestLevel], random, a.levels[coarsestLevel],
                    b.levels[coarsestLevel], input.patchSize, weights);
  Applicator::apply<T>(forward.levels[coarsestLevel], b.levels[coarsestLevel],
                       a.levels[coarsestLevel], input.b.numChannels,
                       input.b.numChannels + input.bPrime.numChannels, input.patchSize);

  // Runs StyLit across the pyramid, starting with the lowest level.
  for (int level = coarsestLevel; level >= 0; level--) {
    printf("\u001B[32m==========================================================\n");
    printf("StyLitCUDA: Calculating pyramid level %d (0 is the finest).\n", level);
    printf("==========================================================\u001B[0m\n");

    // Defines some things to make the code below more legible.
    Image<NNFEntry> &curReverse = reverse.levels[level];
    Image<NNFEntry> &curForward = forward.levels[level];
    Image<T> &curA = a.levels[level];
    Image<T> &curB = b.levels[level];

    for (int optimization = 0; optimization < NUM_OPTIMIZATIONS_PER_LEVEL; optimization++) {
      // Invalidates the forward NNF.
      NNF::invalidate(forward.levels[level]);

      // At this stage, the images (A, B, A', B') should be populated.
      // The reverse NNF should be randomized or prepopulated, depending on the pyramid level.
      // The forward NNF should be entirely empty (invalid).
      const int GIVING_UP_THRESHOLD = 10;
      const float STOPPING_THRESHOLD = 0.95f;
      int pixelsMapped = 0;
      float fractionFilled = 0.f;
      Image<NNFEntry> bestReverseNNF(curReverse.rows, curReverse.cols, 1);
      bestReverseNNF.allocate();
      for (int i = 0; i < GIVING_UP_THRESHOLD; i++) {
        // Improves the NNF.
        PatchMatch::run(curReverse, &curForward, curA, curB, random, input.patchSize,
                        NUM_PATCH_MATCH_ITERATIONS, weights);

        // The reverse NNF after the first iteration is the ideal reverse NNF, since it's completely
        // unaffected by the blacklist. It's copied to bestReverseNNF for later use (since the
        // reverse NNF is upscaled, and we want the best one).
        if (i == 0) {
          cudaMemcpy2D(bestReverseNNF.deviceData, bestReverseNNF.pitch, curReverse.deviceData,
                       curReverse.pitch, curReverse.cols * sizeof(NNFEntry), curReverse.rows,
                       cudaMemcpyDeviceToDevice);
        }
        pixelsMapped += ReverseToForwardNNF::transfer(curReverse, curForward);
        fractionFilled = (float)pixelsMapped / (curForward.rows * curForward.cols);
        if (fractionFilled > STOPPING_THRESHOLD) {
          break;
        }
      }
      printf("StyLitCUDA: NNF is %f percent full (target: %f percent).\n", fractionFilled * 100.f,
             STOPPING_THRESHOLD * 100.f);

      // Generates a forward NNF and uses it to fill the remaining entries in the NNF.
      Image<NNFEntry> tempForward(curForward.rows, curForward.cols, 1);
      tempForward.allocate();
      if (level == coarsestLevel) {
        NNF::randomize(tempForward, random, curB, curA, input.patchSize, weights);
      } else {
        NNF::upscale(forward.levels[level + 1], tempForward, input.patchSize);
        NNF::recalculateErrors(tempForward, curB, curA, input.patchSize, weights);
      }
      PatchMatch::run(tempForward, nullptr, curB, curA, random, input.patchSize,
                      NUM_PATCH_MATCH_ITERATIONS, weights);
      ReverseToForwardNNF::fill(tempForward, curForward);
      tempForward.free();

      if (optimization != NUM_OPTIMIZATIONS_PER_LEVEL - 1) {
        // Updates B' and stays on this pyramid level.
        Applicator::apply<T>(curForward, curB, curA, input.b.numChannels,
                             input.b.numChannels + input.bPrime.numChannels, input.patchSize);
      } else {
        // Prepares for the next pyramid level.
        if (level) {
          // This is a coarse pyramid level.
          // The following needs to happen:
          // 1) The reverse NNF needs to be upscaled to produce the next-finest level's reverse NNF.
          // 2) The next-finest level's reverse NNF needs to be applied to produce B'.
          NNF::upscale(bestReverseNNF, reverse.levels[level - 1], input.patchSize);
          NNF::upscale(curForward, forward.levels[level - 1], input.patchSize);
          Applicator::apply<T>(forward.levels[level - 1], b.levels[level - 1], a.levels[level - 1],
                               input.b.numChannels, input.b.numChannels + input.bPrime.numChannels,
                               input.patchSize);
        } else {
          // This is the finest pyramid level.
          // The final NNF needs to be applied to create an image.
          Applicator::apply<T>(curForward, curB, curA, input.b.numChannels,
                               input.b.numChannels + input.bPrime.numChannels, input.patchSize);
        }
      }
      bestReverseNNF.free();
    }
  }

  // Copies B' back to the caller.
  std::vector<InterfaceImage<T>> bImagesPrime(1);
  bImagesPrime[0] = input.bPrime;
  b.levels[0].retrieveChannels(bImagesPrime, input.b.numChannels);
  weights.deviceFree();
}

template <typename T> Coordinator<T>::~Coordinator() { random.free(); }

template class Coordinator<int>;
template class Coordinator<float>;

void runCoordinator_float(InterfaceInput<float> &input) { Coordinator<float> coordinator(input); }

} /* namespace StyLitCUDA */
