#ifndef STYLITCOORDINATOR_H
#define STYLITCOORDINATOR_H

#include <Configuration/Configuration.h>

/**
 * @brief The StyLitCoordinator class
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class StyLitCoordinator {
public:
  StyLitCoordinator() = default;
  virtual ~StyLitCoordinator() = default;

  /**
   * @brief runStyLit The implementation-specific StyLit implementation is
   * called from here.
   * @return true if StyLit ran successfully; otherwise false
   */
  virtual bool runStyLit(const Configuration &configuration) = 0;
};

#endif // STYLITCOORDINATOR_H
