#ifndef TESTDOWNSCALERWITHIMAGE_H
#define TESTDOWNSCALERWITHIMAGE_H

#include "UnitTest.h"

#include "Algorithm/Downscaler.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "CPU/DownscalerCPU.h"
#include "Configuration/Configuration.h"
#include "Utilities/FloatTools.h"
#include "Utilities/ImageIO.h"

#include <chrono>
#include <iostream>
#include <stdio.h>

// This is a really whacky template thing, but it allows TestDownscaler to test
// arbitrary downscaler implementations.
template <template <typename, unsigned int> class DownscaleImplementation>
class TestDownscalerWithImage : public UnitTest {
public:
  TestDownscalerWithImage() = default;
  bool run() override {
    // Sets up the images that will be used for testing.
    DownscaleImplementation<float, 3> downscaler;
    Configuration configuration;

    // Reads the full image.
    ImageDimensions fullDimensions;
    QString imagePath("./Examples/test_cit.png");
    TEST_ASSERT(ImageIO::getImageDimensions(imagePath, fullDimensions));
    ImageDimensions halfDimensions(fullDimensions.rows / 2,
                                   fullDimensions.cols / 2);
    Image<float, 3> fullImage(fullDimensions);
    TEST_ASSERT(
        ImageIO::readImage<3>(imagePath, fullImage, ImageFormat::RGB, 0));

    // Computes the full image's average.
    float fullSum = 0.f;
    for (int row = 0; row < fullDimensions.rows; row++) {
      for (int col = 0; col < fullDimensions.cols; col++) {
        fullSum += fullImage(row, col)[0] + fullImage(row, col)[1] +
                   fullImage(row, col)[2];
      }
    }
    const float correctAverage = fullSum / (fullDimensions.area() * 3.f);

    // Downscales the image.
    auto start = std::chrono::steady_clock::now();
    Image<float, 3> halfImage(halfDimensions);
    downscaler.downscale(configuration, fullImage, halfImage);
    auto diff = std::chrono::steady_clock::now() - start;
    printf("Downscaling an image (%d, %d) took %f ms.\n", fullDimensions.cols,
           fullDimensions.rows,
           std::chrono::duration<double, std::milli>(diff));

    // Calculates the half image's average and asserts its sameness.
    float halfSum = 0.f;
    for (int row = 0; row < halfDimensions.rows; row++) {
      for (int col = 0; col < halfDimensions.cols; col++) {
        halfSum += halfImage(row, col)[0] + halfImage(row, col)[1] +
                   halfImage(row, col)[2];
      }
    }
    const float experimentalAverage = halfSum / (halfDimensions.area() * 3.f);

    ImageIO::writeImage<3>(
        QString("/home/davidcharatan/Documents/StyLitBin/lol.png"), halfImage,
        ImageFormat::RGB, 0);

    TEST_ASSERT(
        FloatTools::tolerantEquals(correctAverage, experimentalAverage, 0.05f));
    return true;
  }
};

#endif // TESTDOWNSCALERWITHIMAGE_H
