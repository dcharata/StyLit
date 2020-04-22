#ifndef TESTDOWNSCALER_H
#define TESTDOWNSCALER_H

#include "UnitTest.h"

#include "Algorithm/Downscaler.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "CPU/DownscalerCPU.h"
#include "Configuration/Configuration.h"

#include <iostream>

// This is a really whacky template thing, but it allows TestDownscaler to test
// arbitrary downscaler implementations.
template <template <typename, unsigned int> class DownscaleImplementation>
class TestDownscaler : public UnitTest {
public:
  TestDownscaler() = default;
  bool run() override {
    // Sets up the images that will be used for testing.
    DownscaleImplementation<int, 1> downscaler;
    const int VALUE = 128;
    const ImageDimensions fullDimensions(4, 4);
    Image<int, 1> testFull(fullDimensions);
    const ImageDimensions halfDimensions(2, 2);
    Image<int, 1> testHalf(halfDimensions);
    const Configuration configuration;

    // Tests a constant image.
    for (int row = 0; row < fullDimensions.rows; row++) {
      for (int col = 0; col < fullDimensions.cols; col++) {
        testFull(row, col)[0] = VALUE;
      }
    }
    downscaler.downscale(configuration, testFull, testHalf);
    for (int row = 0; row < halfDimensions.rows; row++) {
      for (int col = 0; col < halfDimensions.cols; col++) {
        TEST_ASSERT(testHalf(row, col)[0] == VALUE);
      }
    }

    // Tests a non-constant image.
    for (int row = 0; row < fullDimensions.rows; row += 2) {
      for (int col = 0; col < fullDimensions.cols; col += 2) {
        testFull(row, col)[0] = 1;
        testFull(row, col + 1)[0] = 2;
        testFull(row + 1, col)[0] = 4;
        testFull(row + 1, col + 1)[0] = 5;
      }
    }
    downscaler.downscale(configuration, testFull, testHalf);
    for (int row = 0; row < halfDimensions.rows; row++) {
      for (int col = 0; col < halfDimensions.cols; col++) {
        TEST_ASSERT(testHalf(row, col)[0] == 3);
      }
    }

    // Tests another non-constant image with a different pattern.
    for (int row = 0; row < fullDimensions.rows; row += 2) {
      for (int col = 0; col < fullDimensions.cols; col += 2) {
        testFull(row, col)[0] = 6;
        testFull(row, col + 1)[0] = 4;
        testFull(row + 1, col)[0] = 18;
        testFull(row + 1, col + 1)[0] = 32;
      }
    }
    downscaler.downscale(configuration, testFull, testHalf);
    for (int row = 0; row < halfDimensions.rows; row++) {
      for (int col = 0; col < halfDimensions.cols; col++) {
        TEST_ASSERT(testHalf(row, col)[0] == 15);
      }
    }
    return true;
  }
};

#endif // TESTDOWNSCALER_H
