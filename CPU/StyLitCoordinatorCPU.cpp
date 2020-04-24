#include "StyLitCoordinatorCPU.h"

template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
bool StyLitCoordinatorCPU<T, numGuideChannels, numStyleChannels>::runStyLit(const Configuration &configuration) {
  // implementate main stylit routine here
  return true;
}

// we may want to define numGuideChannels, numStyleChannels in the configuration
template class StyLitCoordinatorCPU<float, 6, 3>;
