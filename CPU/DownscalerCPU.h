#ifndef DOWNSCALERCPU_H
#define DOWNSCALERCPU_H

#include "Algorithm/Downscaler.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Configuration/Configuration.h"

#include <QtGlobal>
#include <stdio.h>

/**
 * @brief The Interpolator struct This is needed because interpolation between 4
 * int values doesn't work with float weights by default. This allows
 * interpolation to be specialized for int and float.
 */
template <typename T, unsigned int numChannels> struct Interpolator {
  static void interpolate(FeatureVector<T, numChannels> &, float,
                          const FeatureVector<T, numChannels> &, float,
                          const FeatureVector<T, numChannels> &, float,
                          const FeatureVector<T, numChannels> &, float,
                          const FeatureVector<T, numChannels> &) {
    // This does nothing since interpolation needs to be specialized for int and
    // float. See the bottom of this file for those specializations.
  }
};

// This is the int specialization of Interpolator.
template <unsigned int numChannels> struct Interpolator<int, numChannels> {
  static void
  interpolate(FeatureVector<int, numChannels> &target, float aWeight,
              const FeatureVector<int, numChannels> &a, float bWeight,
              const FeatureVector<int, numChannels> &b, float cWeight,
              const FeatureVector<int, numChannels> &c, float dWeight,
              const FeatureVector<int, numChannels> &d) {
    // For ints, everything is converted to float before conversion and back to
    // int afterwards.
    const FeatureVector<float, numChannels> aFloat = a.template cast<float>();
    const FeatureVector<float, numChannels> bFloat = b.template cast<float>();
    const FeatureVector<float, numChannels> cFloat = c.template cast<float>();
    const FeatureVector<float, numChannels> dFloat = d.template cast<float>();
    FeatureVector<float, numChannels> result;
    Interpolator<float, numChannels>::interpolate(result, aWeight, aFloat,
                                                  bWeight, bFloat, cWeight,
                                                  cFloat, dWeight, dFloat);
    target = result.template cast<int>();
  }
};

// This is the float specialization of Interpolator.
template <unsigned int numChannels> struct Interpolator<float, numChannels> {
  static void
  interpolate(FeatureVector<float, numChannels> &target, float aWeight,
              const FeatureVector<float, numChannels> &a, float bWeight,
              const FeatureVector<float, numChannels> &b, float cWeight,
              const FeatureVector<float, numChannels> &c, float dWeight,
              const FeatureVector<float, numChannels> &d) {
    // For floats, interpolation is easy.
    target = a * aWeight + b * bWeight + c * cWeight + d * dWeight;
  }
};

template <typename T, unsigned int numChannels>
class DownscalerCPU : public Downscaler<T, numChannels> {
public:
  DownscalerCPU() = default;
  virtual ~DownscalerCPU() = default;

protected:
  /**
   * @brief sampleBilinear Does bilinear sampling of the image at the given
   * float coordiantes.
   * @param image the image to sample
   * @param row the row to sample at
   * @param col the column to sample at
   * @param target the FeatureVector to populate the sample with
   */
  void sampleBilinear(const Image<T, numChannels> &image, float row, float col,
                      FeatureVector<T, numChannels> &target) const {
    // Calculates some things needed for sampling.
    // Hopefully this makes things a bit easier to read and understand.
    const int rowFloor = qBound(0, int(row), image.dimensions.row);
    const int colFloor = qBound(0, int(col), image.dimensions.col);
    const int rowCeil = qBound(0, rowFloor + 1, image.dimensions.row);
    const int colCeil = qBound(0, colFloor + 1, image.dimensions.col);
    const float rowRemainderForCeil = row - rowFloor;
    const float rowRemainderForFloor = 1.f - rowRemainderForCeil;
    const float colRemainderForCeil = col - colFloor;
    const float colRemainderForFloor = 1.f - colRemainderForCeil;

    // Calculates the samples.
    const float topLeftWeight = rowRemainderForFloor * colRemainderForFloor;
    const FeatureVector<T, numChannels> topLeft =
        image.getConstPixel(rowFloor, colFloor);

    const float topRightWeight = rowRemainderForFloor * colRemainderForCeil;
    const FeatureVector<T, numChannels> topRight =
        image.getConstPixel(rowFloor, colCeil);

    const float bottomLeftWeight = rowRemainderForCeil * colRemainderForFloor;
    const FeatureVector<T, numChannels> bottomLeft =
        image.getConstPixel(rowCeil, colFloor);

    const float bottomRightWeight = rowRemainderForCeil * colRemainderForCeil;
    const FeatureVector<T, numChannels> bottomRight =
        image.getConstPixel(rowCeil, colCeil);

    Interpolator<T, numChannels>::interpolate(
        target, topLeftWeight, topLeft, topRightWeight, topRight,
        bottomLeftWeight, bottomLeft, bottomRightWeight, bottomRight);
  }

  /**
   * @brief implementationOfDownscale Downscales the image full to half the
   * resolution, storing the result in the image half. Uses bilinear
   * interpolation based on EBSynth's.
   * https://github.com/jamriska/ebsynth
   * @param configuration the configuration StyLit is running
   * @param full the image to downscale from
   * @param half the image to downscale to
   * @return true if downscaling succeeds; otherwise false
   */
  bool implementationOfDownscale(const Configuration &,
                                 const Image<T, numChannels> &full,
                                 Image<T, numChannels> &half) override {
    const float colScale =
        float(full.dimensions.cols) / float(half.dimensions.cols);
    const float rowScale =
        float(full.dimensions.rows) / float(half.dimensions.rows);
    for (int row = 0; row < half.dimensions.rows; row++) {
      for (int col = 0; col < half.dimensions.cols; col++) {
        sampleBilinear(full, row * rowScale + 0.5f, col * colScale + 0.5f,
                       half(row, col));
      }
    }
    return true;
  }
};

#endif // DOWNSCALERCPU_H
