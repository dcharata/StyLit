#include "TestImageIOWrite.h"

#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Utilities/FloatTools.h"
#include "Utilities/ImageFormat.h"
#include "Utilities/ImageIO.h"

#include <QFile>

bool TestImageIOWrite::run() {
  // Generates the image to write.
  QString path("./Examples/write_test_image.png");
  const int ROWS = 12;
  const int COLS = 10;
  const ImageDimensions dimensions(ROWS, COLS);
  const float TOLERANCE = 1.f / 256.f;

  // Creates a random 4-channel image.
  srand(2954);
  Image<float, 4> testImage(dimensions);
  for (int row = 0; row < ROWS; row++) {
    for (int col = 0; col < COLS; col++) {
      testImage(row, col)[0] = float(rand() % 256) / 256.f;
      testImage(row, col)[1] = float(rand() % 256) / 256.f;
      testImage(row, col)[2] = float(rand() % 256) / 256.f;
      testImage(row, col)[3] = float(rand() % 256) / 256.f;
    }
  }

  // Writes all 4 channels as RGBA, then reads them back and confirms that the
  // results are the same.
  {
    TEST_ASSERT(ImageIO::writeImage<4>(path, testImage, ImageFormat::RGBA, 0));
    Image<float, 4> retrievedTestImage(dimensions);
    TEST_ASSERT(
        ImageIO::readImage(path, retrievedTestImage, ImageFormat::RGBA, 0));
    for (int row = 0; row < ROWS; row++) {
      for (int col = 0; col < COLS; col++) {
        for (int channel = 0; channel < 4; channel++) {
          const float expected = testImage(row, col)[channel];
          const float received = retrievedTestImage(row, col)[channel];
          TEST_ASSERT(
              FloatTools::tolerantEquals(expected, received, TOLERANCE));
        }
      }
    }
  }

  // Writes the last 3 channels as RGB, then reads them back and confirms that
  // the contents are the same.
  {
    TEST_ASSERT(ImageIO::writeImage<4>(path, testImage, ImageFormat::RGB, 1));
    Image<float, 4> retrievedTestImage(dimensions);
    TEST_ASSERT(
        ImageIO::readImage(path, retrievedTestImage, ImageFormat::RGB, 1));
    for (int row = 0; row < ROWS; row++) {
      for (int col = 0; col < COLS; col++) {
        for (int channel = 1; channel < 4; channel++) {
          const float expected = testImage(row, col)[channel];
          const float received = retrievedTestImage(row, col)[channel];
          TEST_ASSERT(
              FloatTools::tolerantEquals(expected, received, TOLERANCE));
        }
      }
    }
  }

  // Writes the second channel as BW, then reads it back and confirms that the
  // contents are the same.
  {
    TEST_ASSERT(ImageIO::writeImage<4>(path, testImage, ImageFormat::BW, 1));
    Image<float, 4> retrievedTestImage(dimensions);
    TEST_ASSERT(
        ImageIO::readImage(path, retrievedTestImage, ImageFormat::BW, 1));
    for (int row = 0; row < ROWS; row++) {
      for (int col = 0; col < COLS; col++) {
        const float expected = testImage(row, col)[1];
        const float received = retrievedTestImage(row, col)[1];
        TEST_ASSERT(FloatTools::tolerantEquals(expected, received, TOLERANCE));
      }
    }
  }

  // Writes the third and fourth channels as BWA, then reads them back and
  // confirms that the contents are the same.
  {
    TEST_ASSERT(ImageIO::writeImage<4>(path, testImage, ImageFormat::BWA, 2));
    Image<float, 4> retrievedTestImage(dimensions);
    TEST_ASSERT(
        ImageIO::readImage(path, retrievedTestImage, ImageFormat::BWA, 2));
    for (int row = 0; row < ROWS; row++) {
      for (int col = 0; col < COLS; col++) {
        for (int channel = 2; channel < 4; channel++) {
          const float expected = testImage(row, col)[channel];
          const float received = retrievedTestImage(row, col)[channel];
          TEST_ASSERT(
              FloatTools::tolerantEquals(expected, received, TOLERANCE));
        }
      }
    }
  }

  // Removes the test file.
  QFile file(path);
  TEST_ASSERT(file.remove());
  return true;
}
