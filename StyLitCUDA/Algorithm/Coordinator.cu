#include "Coordinator.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

namespace StyLitCUDA {

template <typename T>
__global__  void test(PyramidImagePitch<T> test) {
  printf("hi from the GPU I guess\n");
}

template <typename T> Coordinator<T>::Coordinator(InterfaceInput<T> &input) : input(input), a(input.a.rows, input.a.cols, input.a.numChannels + input.aPrime.numChannels, input.numLevels), b(input.b.rows, input.b.cols, input.b.numChannels + input.bPrime.numChannels, input.numLevels) {
  // A and B are initialized here since putting them in the initializer lists would be confusing.
  a.allocate();
  b.allocate();

  // Loads the images into A and B.
  // A contains both A and A'.
  std::vector<InterfaceImage<T>> aImages(2);
  aImages[0] = input.a;
  aImages[1] = input.aPrime;
  a.populateTopLevel(aImages, 0);

  // B contains only B (since B' is filled in by StyLit).
  std::vector<InterfaceImage<T>> bImages(1);
  bImages[0] = input.b;
  b.populateTopLevel(bImages, 0);

  test<T><<<1, 1>>>(a);
}

template <typename T> Coordinator<T>::~Coordinator() {
  a.free();
  b.free();
}

template class Coordinator<int>;
template class Coordinator<float>;

void runCoordinator_float(InterfaceInput<float> &input) {
  Coordinator<float> coordinator(input);
}

} /* namespace StyLitCUDA */
