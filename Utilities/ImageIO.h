#ifndef IMAGEREADER_H
#define IMAGEREADER_H

#include <QImage>
#include <QRgb>
#include <QString>

#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "ImageFormat.h"
#include "ImageFormatTools.h"

namespace ImageIO {

/**
 * @brief getImageDimensions Fills dimensions with the dimensions of the image
 * at path. This is needed because an image's size needs to be known when it's
 * created. Reading one extra source image and one extra target image isn't
 * ideal, but changing readImage (or Image itself) to avoid this is sort of
 * complicated. I tried doing so and gave up after a while.
 * @param path the image to read
 * @param dimensions the image's dimensions
 * @return true if dimensions could be read; otherwise false
 */
bool getImageDimensions(const QString &path, ImageDimensions &dimensions) {
  QImage qImage(path);
  if (qImage.isNull()) {
    return false;
  }
  dimensions.rows = qImage.height();
  dimensions.cols = qImage.width();
  return true;
}

/**
 * @brief readImage Reads an image into the specified FeatureVector image.
 * @param path the path to read the image from
 * @param image the FeatureVector image to populate
 * @param imageFormat the ImageFormat to use to interpret the image
 * @param startingChannel the first channel in image to populate
 * @return true if reading the image succeeds; otherwise false
 */
template <unsigned int numChannels>
bool readImage(const QString &path, Image<float, numChannels> &image,
               const ImageFormat &imageFormat, int startingChannel) {
  // Asserts that the range of channels to read is valid.
  Q_ASSERT(startingChannel >= 0 &&
           startingChannel + ImageFormatTools::numChannels(imageFormat) <=
               int(numChannels));

  // Reads the QImage.
  QImage qImage(path);
  if (qImage.isNull()) {
    return false;
  }

  // If allocate isn't asserted and the FeatureVector image's dimensions don't
  // match the QImage's dimensions, reading fails.
  if (image.dimensions.rows != qImage.height() ||
      image.dimensions.cols != qImage.width()) {
    return false;
  }

  // Does the actual reading of the pixels.
  for (int row = 0; row < image.dimensions.rows; row++) {
    for (int col = 0; col < image.dimensions.cols; col++) {
      // Extracts the pixel information.
      const QRgb pixel = qImage.pixel(col, row);
      const float red = float(qRed(pixel)) / 255.f;
      const float green = float(qGreen(pixel)) / 255.f;
      const float blue = float(qBlue(pixel)) / 255.f;
      const float alpha = float(qAlpha(pixel)) / 255.f;

      // Puts the pixel information into the image.
      FeatureVector<float, numChannels> &featureVector = image(row, col);
      switch (imageFormat) {
      case ImageFormat::BW:
        featureVector[startingChannel] = (red + green + blue) / 3.f;
        break;
      case ImageFormat::BWA:
        featureVector[startingChannel] = (red + green + blue) / 3.f;
        featureVector[startingChannel + 1] = alpha;
        break;
      case ImageFormat::RGB:
        featureVector[startingChannel] = red;
        featureVector[startingChannel + 1] = green;
        featureVector[startingChannel + 2] = blue;
        break;
      case ImageFormat::RGBA:
        featureVector[startingChannel] = red;
        featureVector[startingChannel + 1] = green;
        featureVector[startingChannel + 2] = blue;
        featureVector[startingChannel + 3] = alpha;
        break;
      default:
        // If the image format is unrecognized, reading fails.
        return false;
      }
    }
  }
  return true;
}

template <unsigned int numChannels>
bool writeImage(const QString &path, Image<float, numChannels> &image,
                const ImageFormat &imageFormat, int startingChannel) {
  // Asserts that the range of channels to write is valid.
  Q_ASSERT(startingChannel >= 0 &&
           startingChannel + ImageFormatTools::numChannels(imageFormat) <
               numChannels);

  // TODO
}
}; // namespace ImageIO

#endif // IMAGEREADER_H
