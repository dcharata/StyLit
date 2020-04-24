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

#endif // STYLITCOORDINATORCPU_H
