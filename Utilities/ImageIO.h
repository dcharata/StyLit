#ifndef IMAGEREADER_H
#define IMAGEREADER_H

#include <QImage>
#include <QRgb>
#include <QString>
#include <iostream>

#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Algorithm/ImagePair.h"
#include "Algorithm/PyramidLevel.h"
#include "Configuration/Configuration.h"
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
bool getImageDimensions(const QString &path, ImageDimensions &dimensions);

/**
 * @brief floatToChar Converts a float in the range [0, 1] to a char.
 * @param value the float value
 * @return the char value [0, 255]
 */
int floatToChar(float value);

/**
 * @brief charToFloat Converts an int in the range [0, 255] to a float in the
 * range [0, 1].
 * @param value the int value
 * @return the float value [0, 1]
 */
float charToFloat(int value);

/**
 * @brief floatsToPixel Converts floating point RGBA values to an integer that
 * encodes each channel in one char (0-255).
 * @param r red
 * @param g green
 * @param b blue
 * @param a alpha
 * @return an integer that stores RGBA like QRgb
 */
int floatsToPixel(float r, float g, float b, float a);

/**
 * @brief pixelToFloats Converts a pixel encoded in qRgb format to floats.
 * @param pixel the pixel in qRgb format (the order is ARGB)
 * @param r red
 * @param g green
 * @param b blue
 * @param a alpha
 */
void pixelToFloats(int pixel, float &r, float &g, float &b, float &a);

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
           startingChannel + ImageFormatTools::numChannels(imageFormat) <= int(numChannels));

  // Reads the QImage.
  QImage qImage(path);
  if (qImage.isNull()) {
    return false;
  }

  // If allocate isn't asserted and the FeatureVector image's dimensions don't
  // match the QImage's dimensions, reading fails.
  if (image.dimensions.rows != qImage.height() || image.dimensions.cols != qImage.width()) {
    return false;
  }

  // Does the actual reading of the pixels.
  for (int row = 0; row < image.dimensions.rows; row++) {
    for (int col = 0; col < image.dimensions.cols; col++) {
      // Extracts the pixel information.
      const QRgb pixel = qImage.pixel(col, row);
      float red, green, blue, alpha;
      pixelToFloats(pixel, red, green, blue, alpha);

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
        std::cerr << "Unrecognized format." << std::endl;
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
           startingChannel + ImageFormatTools::numChannels(imageFormat) <= int(numChannels));

  // Determines the save format.
  QImage::Format format;
  switch (imageFormat) {
  case ImageFormat::BW:
  case ImageFormat::RGB:
    format = QImage::Format_RGB888;
    break;
  case ImageFormat::BWA:
  case ImageFormat::RGBA:
    format = QImage::Format_RGBA8888;
    break;
  }

  // Creates the image that will be written.
  const QSize size(image.dimensions.cols, image.dimensions.rows);
  QImage qImage(size, format);

  // Populates the QImage.
  for (int row = 0; row < image.dimensions.rows; row++) {
    for (int col = 0; col < image.dimensions.cols; col++) {
      FeatureVector<float, numChannels> &featureVector = image(row, col);

      // Fills in the pixel based on the image format.
      QRgb pixel;
      switch (imageFormat) {
      case ImageFormat::BW: {
        const float intensity = featureVector[startingChannel];
        pixel = floatsToPixel(intensity, intensity, intensity, 1.f);
        break;
      }
      case ImageFormat::BWA: {
        const float intensity = featureVector[startingChannel];
        pixel = floatsToPixel(intensity, intensity, intensity, featureVector[startingChannel + 1]);
        break;
      }
      case ImageFormat::RGB:
        pixel = floatsToPixel(featureVector[startingChannel], featureVector[startingChannel + 1],
                              featureVector[startingChannel + 2], 1.f);
        break;
      case ImageFormat::RGBA:
        pixel =
            floatsToPixel(featureVector[startingChannel], featureVector[startingChannel + 1],
                          featureVector[startingChannel + 2], featureVector[startingChannel + 3]);
        break;
      default:
        // If the image format is unrecognized, reading fails.
        std::cerr << "Unrecognized format." << std::endl;
        return false;
      }
      qImage.setPixel(col, row, pixel);
    }
  }

  // Writes the image to disk.
  return qImage.save(path);
}

template <unsigned int numChannels>
bool readFeatureVectorImage(Image<float, numChannels> &image, const std::vector<QString> &paths,
                            const std::vector<ImageFormat> &formats) {
  int numFilledChannels = 0;
  for (unsigned int i = 0; i < paths.size(); i++) {
    const ImageFormat &format = formats[i];
    const QString &path = paths[i];
    const int formatChannels = ImageFormatTools::numChannels(format);
    if (!ImageIO::readImage(path, image, format, numFilledChannels)) {
      std::cerr << "Failed image read." << std::endl;
      return false;
    }
    numFilledChannels += formatChannels;
  }
  return numFilledChannels == numChannels;
}

/**
 * @brief readPyramidLevel Populates the given pyramid level with the images
 * specified in the configuration file.
 * @param configuration the configuration file's contents
 * @param pyramidLevel the PyramidLevel to populate
 * @return true if reading succeeds; otherwise false
 */
template <unsigned int numGuideChannels, unsigned int numStyleChannels>
bool readPyramidLevel(const Configuration &configuration,
                      PyramidLevel<float, numGuideChannels, numStyleChannels> &pyramidLevel) {
  return readFeatureVectorImage(pyramidLevel.guide.source, configuration.sourceGuideImagePaths,
                                configuration.guideImageFormats) &&
         readFeatureVectorImage(pyramidLevel.guide.target, configuration.targetGuideImagePaths,
                                configuration.guideImageFormats) &&
         readFeatureVectorImage(pyramidLevel.style.source, configuration.sourceStyleImagePaths,
                                configuration.styleImageFormats);
}

}; // namespace ImageIO

#endif // IMAGEREADER_H
