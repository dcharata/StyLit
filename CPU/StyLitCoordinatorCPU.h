#ifndef STYLITCOORDINATORCPU_H
#define STYLITCOORDINATORCPU_H

#include "Algorithm/StyLitCoordinator.h"
#include "NNFGeneratorCPU.h"

template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class StyLitCoordinatorCPU : public StyLitCoordinator<T, numGuideChannels, numStyleChannels>
{
public:
  StyLitCoordinatorCPU() = default;
  ~StyLitCoordinatorCPU() = default;

  /**
   * @brief runStyLit The implementation-specific StyLit implementation is
   * called from here.
   * @return true if StyLit ran successfully; otherwise false
   */
  virtual bool runStyLit(const Configuration &configuration) override;
};


/**
* template class implementation
* why not in .cpp - to avoid explicitly instantiation in .cpp and fixing the parameters
* Ref here: in answer 2
* https://stackoverflow.com/questions/8752837/undefined-reference-to-template-class-constructor
*/
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
bool StyLitCoordinatorCPU<T, numGuideChannels, numStyleChannels>::runStyLit(const Configuration &configuration) {
  // implementate main stylit routine here
  return true;
}

#endif // STYLITCOORDINATORCPU_H
