#include "Coordinator.cuh"

#include <vector>

namespace StyLitCUDA {

template <typename T> Coordinator<T>::Coordinator(InterfaceInput<T> &input) : input(input) {
  // A and B are initialized here since putting them in the initializer lists would be confusing.
  a = std::make_unique<PyramidImagePitch<T>>(
      input.a.rows, input.a.cols, input.a.numChannels + input.aPrime.numChannels, input.numLevels);
  b = std::make_unique<PyramidImagePitch<T>>(
      input.b.rows, input.b.cols, input.b.numChannels + input.bPrime.numChannels, input.numLevels);
  a->allocate();
  //b->allocate();

  // Loads the images into A and B.
  //std::vector<InterfaceImage<T>> aImages(2);
  //aImages.push_back(input.a);
  //aImages.push_back(input.aPrime);
  //a->populateTopLevel(aImages, 0);
}

template <typename T> Coordinator<T>::~Coordinator() {
  a->free();
  //b->free();
}

template class Coordinator<int>;
template class Coordinator<float>;

void runCoordinator_float(InterfaceInput<float> &input) {
  Coordinator<float> coordinator(input);
}

} /* namespace StyLitCUDA */
