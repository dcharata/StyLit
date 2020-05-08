#ifndef STYLITCOORDINATOR_H
#define STYLITCOORDINATOR_H

#include <Configuration/Configuration.h>

/**
 * @brief The StyLitCoordinatorBase class This is needed so that an array of
 * pointers to different templated implementations can be created.
 */
class StyLitCoordinatorBase {
public:
  StyLitCoordinatorBase() = default;
  virtual ~StyLitCoordinatorBase() = default;

  /**
   * @brief runStyLit The implementation-specific StyLit implementation is
   * called from here.
   * @return true if StyLit ran successfully; otherwise false
   */
  virtual bool runStyLit(const Configuration &configuration) = 0;
};

/**
 * @brief The StyLitCoordinator class
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class StyLitCoordinator : public StyLitCoordinatorBase {
public:
  StyLitCoordinator() = default;
  virtual ~StyLitCoordinator() = default;
};

#endif // STYLITCOORDINATOR_H
