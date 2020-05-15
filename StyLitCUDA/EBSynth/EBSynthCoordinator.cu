#include "EBSynthCoordinator.cuh"

#include "../Algorithm/Applicator.cuh"
#include "../Algorithm/Downscaler.cuh"
#include "../Algorithm/NNF.cuh"
#include "../Algorithm/RandomInitializer.cuh"
#include "../Algorithm/ReverseToForwardNNF.cuh"
#include "../Utilities/Image.cuh"
#include "../Utilities/Utilities.cuh"
#include "../Utilities/Vec.cuh"
#include "EBSynthPatchMatch.cuh"
#include "Omega.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

namespace StyLitCUDA {

template <typename T>
EBSynthCoordinator<T>::EBSynthCoordinator(InterfaceInput<T> &input)
    : input(input), a(input.a.rows, input.a.cols, input.a.numChannels + input.aPrime.numChannels,
                      input.numLevels),
      b(input.b.rows, input.b.cols, input.b.numChannels + input.bPrime.numChannels,
        input.numLevels),
      omegas(input.a.rows, input.a.cols, 1, input.numLevels),
      random(std::max(input.a.rows, input.b.rows), std::max(input.a.cols, input.b.cols), 1),
      forward(input.b.rows, input.b.cols, 1, input.numLevels) {
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
  NNF::randomize<T>(forward.levels[coarsestLevel], random, b.levels[coarsestLevel],
                    a.levels[coarsestLevel], input.patchSize, weights);
  Applicator::apply<T>(forward.levels[coarsestLevel], b.levels[coarsestLevel],
                       a.levels[coarsestLevel], input.b.numChannels,
                       input.b.numChannels + input.bPrime.numChannels, input.patchSize);
  NNF::recalculateErrors(forward.levels[coarsestLevel], b.levels[coarsestLevel], a.levels[coarsestLevel], input.patchSize, weights);

  // Initializes the omega values at the lowest pyramid level.
  Omega::initialize(forward.levels[coarsestLevel], omegas.levels[coarsestLevel], input.patchSize);

  // Runs StyLit across the pyramid, starting with the lowest level.
  for (int level = coarsestLevel; level >= 0; level--) {
    printf("\u001B[32m==========================================================\n");
    printf("EBSynthCUDA: Calculating pyramid level %d (0 is the finest).\n", level);
    printf("==========================================================\u001B[0m\n");

    // Defines some things to make the code below more legible.
    Image<NNFEntry> &curForward = forward.levels[level];
    Image<float> &curOmegas = omegas.levels[level];
    Image<T> &curA = a.levels[level];
    Image<T> &curB = b.levels[level];

    for (int optimization = 0; optimization < NUM_OPTIMIZATIONS_PER_LEVEL; optimization++) {
      // If not at the coarsest level, upscales the previous level's NNF.
      if (level != coarsestLevel && optimization == 0) {
        NNF::upscale(forward.levels[level + 1], curForward, input.patchSize);
        Applicator::apply<T>(curForward, curB, curA, input.b.numChannels,
                             input.b.numChannels + input.bPrime.numChannels, input.patchSize);
        NNF::recalculateErrors(curForward, curB, curA, input.patchSize, weights);
        Omega::initialize(curForward, curOmegas, input.patchSize);
      }

      // Runs PatchMatch.
      EBSynthPatchMatch::run(curForward, curOmegas, curB, curA, random, input.patchSize,
                             NUM_PATCH_MATCH_ITERATIONS, weights);

      // Updates B' if this isn't the last optimization iteration or this is the last level.
      if (optimization != NUM_OPTIMIZATIONS_PER_LEVEL - 1 || !level) {
        Applicator::apply<T>(curForward, curB, curA, input.b.numChannels,
                             input.b.numChannels + input.bPrime.numChannels, input.patchSize);
      }
    }
  }

  // Copies B' back to the caller.
  std::vector<InterfaceImage<T>> bImagesPrime(1);
  bImagesPrime[0] = input.bPrime;
  b.levels[0].retrieveChannels(bImagesPrime, input.b.numChannels);
  weights.deviceFree();
}

template <typename T> EBSynthCoordinator<T>::~EBSynthCoordinator() { random.free(); }

template class EBSynthCoordinator<int>;
template class EBSynthCoordinator<float>;

void runEBSynthCoordinator_float(InterfaceInput<float> &input) {
  EBSynthCoordinator<float> EBSynthCoordinator(input);
}

} /* namespace StyLitCUDA */
