#include "Coordinator.cuh"

#include "../Utilities/ImagePitch.cuh"
#include "../Utilities/Utilities.cuh"
#include "Downscaler.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

namespace StyLitCUDA {

template <typename T>
Coordinator<T>::Coordinator(InterfaceInput<T> &input)
    : input(input), a(input.a.rows, input.a.cols, input.a.numChannels + input.aPrime.numChannels,
                      input.numLevels),
      b(input.b.rows, input.b.cols, input.b.numChannels + input.bPrime.numChannels,
        input.numLevels) {
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
    Downscaler::downscale(a.levels[level], a.levels[level + 1]);
  }
  for (int level = 0; level < b.levels.size() - 1; level++) {
    Downscaler::downscale(b.levels[level], b.levels[level + 1]);
  }

  // Copies B' back to the caller.
  std::vector<InterfaceImage<T>> bImagesPrime(1);
  bImagesPrime[0] = input.bPrime;
  a.levels[1].retrieveChannels(bImagesPrime, 0);
}

template <typename T> Coordinator<T>::~Coordinator() {}

template class Coordinator<int>;
template class Coordinator<float>;

void runCoordinator_float(InterfaceInput<float> &input) { Coordinator<float> coordinator(input); }

} /* namespace StyLitCUDA */
