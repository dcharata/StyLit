#ifndef DOWNSCALER_H
#define DOWNSCALER_H

#include "Configuration/Configuration.h"
#include "Image.h"
#include "ImageDimensions.h"

#include <QtGlobal>

/**
 * @brief The Downscaler class This is the interface for the
 * implementation-specific image downscaling function, which is used in the
 * generation of image pyramids.
 */
template <typename T, unsigned int numChannels> class Downscaler {
public:
  Downscaler() = default;
  virtual ~Downscaler() = default;

  /**
   * @brief downscale This is a wrapper around implementationOfDownscale. It
   * asserts that the image dimensions make sense before calling the
   * implementation-specific image downscaler.
   * @param configuration the configuration StyLit is running
   * @param full the image to downscale from
   * @param half the image to downscale to
   * @return true if downscaling succeeds; otherwise false
   */
  bool downscale(const Configuration &configuration,
                 const Image<T, numChannels> &full,
                 Image<T, numChannels> &half) {
    Q_ASSERT(half.dimensions.halfTheSizeOf(full.dimensions));
    return implementationOfDownscale(configuration, full, half);
  }

protected:
  /**
   * @brief implementationOfDownscale Downscales the image full to half the
   * resolution, storing the result in the image half.
   * @param configuration the configuration StyLit is running
   * @param full the image to downscale from
   * @param half the image to downscale to
   * @return true if downscaling succeeds; otherwise false
   */
  virtual bool implementationOfDownscale(const Configuration &configuration,
                                         const Image<T, numChannels> &full,
                                         Image<T, numChannels> &half) = 0;
};

#endif // DOWNSCALER_H
