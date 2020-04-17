#ifndef DOWNSCALERCPU_H
#define DOWNSCALERCPU_H

#include "Configuration/Configuration.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Algorithm/Downscaler.h"

#include <QtGlobal>

/**
 * @brief The Downscaler class This is the interface for the
 * implementation-specific image downscaling function, which is used in the
 * generation of image pyramids.
 */
template <typename T, unsigned int numChannels>
class DownscalerCPU : public Downscaler<T, numChannels> {
public:
  DownscalerCPU() = default;
  virtual ~DownscalerCPU() = default;

  /**
   * @brief downscale This is a wrapper around implementationOfDownscale. It
   * asserts that the image dimensions make sense before calling the
   * implementation-specific image downscaler.
   * @param configuration the configuration StyLit is running
   * @param full the image to downscale from
   * @param half the image to downscale to
   * @return true if downscaling succeeds; otherwise false
   */
  bool downscaleCPU(const Configuration &configuration,
                 const Image<T, numChannels> &full,
                 Image<T, numChannels> &half) {
    Q_ASSERT(half.dimensions.halfTheSizeOf(full));
    return implementationOfDownscaleCPU(configuration, full, half);
  }

protected:
  // Taken from ebsynth
  FeatureVector<T, numChannels> bilinearInterpolation(const Image<T, numChannels> &I,
                                                      float i, float j) {

      const int ii = (int)i;
      const int ij = (int)j;
      const float s = i - ii;
      const float t = j - ij;

      return FeatureVector<T, numChannels>((1.0f - s) * (1.0f - t) * I.data[I.dimensions.cols * qBound(0,ii,I.dimensions.rows) + qBound(0,ij,I.dimensions.cols)] +
              s * t * I.data[I.dimensions.cols * qBound(0,ii + 1,I.dimensions.rows) + qBound(0,ij + 1,I.dimensions.cols)] +
              s * (1.0f - t) * I.data[I.dimensions.cols * qBound(0,ii + 1,I.dimensions.rows) + qBound(0,ij,I.dimensions.cols)] +
              t * (1.0f - s) * I.data[I.dimensions.cols * qBound(0,ii,I.dimensions.rows) + qBound(0,ij + 1,I.dimensions.cols)]);
  }

  /**
   * @brief implementationOfDownscale Downscales the image full to half the
   * resolution, storing the result in the image half.
   * @param configuration the configuration StyLit is running
   * @param full the image to downscale from
   * @param half the image to downscale to
   * @return true if downscaling succeeds; otherwise false
   */
  virtual bool implementationOfDownscaleCPU(const Configuration &configuration,
                                         const Image<T, numChannels> &full,
                                            Image<T, numChannels> &half) {

      const float sj = (float)full.dimensions.cols / half.dimensions.cols;
      const float si = (float)full.dimensions.rows / half.dimensions.rows;

      for (int i = 0; i < full.dimensions.rows; i++) {
          for (int j = 0; j < full.dimensions.cols; j++) {
              half.data[half.dimensions.cols * i + j] = bilinearInterpolation(full, si * (float)i, sj * (float)j);
          }
      }
      return 1;
  }
};

#endif // DOWNSCALERCPU_H
