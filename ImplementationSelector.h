#ifndef IMPLEMENTATIONSELECTOR_H
#define IMPLEMENTATIONSELECTOR_H

#include "Configuration/Configuration.h"

namespace ImplementationSelector {
/**
 * @brief runWithConfiguration Picks the correct implementation of StyLit based
 * on the configuration file, then runs it.
 * @param configuration the configuration to run
 * @return true if running StyLit succeeds; otherwise false
 */
bool runWithConfiguration(const Configuration &configuration);

/**
 * @brief runCPU Picks the correct template version of the CPU implementation.
 * @param configuration the configuration
 * @return true if running StyLit succeeds; otherwise false
 */
bool runCPU(const Configuration &configuration);

/**
 * @brief runCUDA Picks the correct template version of the CUDA implementation.
 * @param configuration the configuration
 * @return true if running StyLit succeeds; otherwise false
 */
bool runCUDA(const Configuration &configuration);

}; // namespace ImplementationSelector

#endif // IMPLEMENTATIONSELECTOR_H
