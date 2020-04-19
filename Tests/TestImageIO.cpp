#include "TestImageIO.h"

#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Utilities/FloatTools.h"
#include "Utilities/ImageFormat.h"
#include "Utilities/ImageIO.h"

bool TestImageIO::run() {
  // These are the constants used to test basic RGB image reading.
  QString path("./Examples/test_10x12_peach.png");
  const float expectedRed = 255.f / 255.f;
  const float expectedGreen = 128.f / 255.f;
  const float expectedBlue = 64.f / 255.f;
  const float expectedGrayscale =
      (expectedRed + expectedGreen + expectedBlue) / 3.f;
  const float expectedAlpha = 128.f / 255.f;

  // Gets the image dimensions.
  ImageDimensions dimensions;
  TEST_ASSERT(ImageIO::getImageDimensions(path, dimensions));
  TEST_ASSERT(dimensions.rows == 12);
  TEST_ASSERT(dimensions.cols == 10);

  // Attempts to read the image as RGB.
  Image<float, 3> imageAsRGB(dimensions);
  TEST_ASSERT(ImageIO::readImage<3>("./Examples/test_10x12_peach.png",
                                    imageAsRGB, ImageFormat::RGB, 0));

  // Confirms that all the pixels have the correct values.
  for (int row = 0; row < dimensions.rows; row++) {
    for (int col = 0; col < dimensions.cols; col++) {
      TEST_ASSERT(
          FloatTools::tolerantEquals(expectedRed, imageAsRGB(row, col)[0]));
      TEST_ASSERT(
          FloatTools::tolerantEquals(expectedGreen, imageAsRGB(row, col)[1]));
      TEST_ASSERT(
          FloatTools::tolerantEquals(expectedBlue, imageAsRGB(row, col)[2]));
    }
  }

  // Attempts to read the image as RGBA.
  Image<float, 4> imageAsRGBA(dimensions);
  TEST_ASSERT(ImageIO::readImage<4>("./Examples/test_10x12_peach.png",
                                    imageAsRGBA, ImageFormat::RGBA, 0));

  // Confirms that all the pixels have the correct values.
  for (int row = 0; row < dimensions.rows; row++) {
    for (int col = 0; col < dimensions.cols; col++) {
      TEST_ASSERT(
          FloatTools::tolerantEquals(expectedRed, imageAsRGBA(row, col)[0]));
      TEST_ASSERT(
          FloatTools::tolerantEquals(expectedGreen, imageAsRGBA(row, col)[1]));
      TEST_ASSERT(
          FloatTools::tolerantEquals(expectedBlue, imageAsRGBA(row, col)[2]));
      TEST_ASSERT(
          FloatTools::tolerantEquals(expectedAlpha, imageAsRGBA(row, col)[3]));
    }
  }

  // Attempts to read the image as BW.
  Image<float, 1> imageAsBW(dimensions);
  TEST_ASSERT(ImageIO::readImage<1>("./Examples/test_10x12_peach.png",
                                    imageAsBW, ImageFormat::BW, 0));

  // Confirms that all the pixels have the correct values.
  for (int row = 0; row < dimensions.rows; row++) {
    for (int col = 0; col < dimensions.cols; col++) {
      TEST_ASSERT(FloatTools::tolerantEquals(expectedGrayscale,
                                             imageAsBW(row, col)[0]));
    }
  }

  // Attempts to read the image as BWA.
  Image<float, 2> imageAsBWA(dimensions);
  TEST_ASSERT(ImageIO::readImage<2>("./Examples/test_10x12_peach.png",
                                    imageAsBWA, ImageFormat::BWA, 0));

  // Confirms that all the pixels have the correct values.
  for (int row = 0; row < dimensions.rows; row++) {
    for (int col = 0; col < dimensions.cols; col++) {
      TEST_ASSERT(FloatTools::tolerantEquals(expectedGrayscale,
                                             imageAsBWA(row, col)[0]));
      TEST_ASSERT(
          FloatTools::tolerantEquals(expectedAlpha, imageAsBWA(row, col)[1]));
    }
  }
  return true;
}
