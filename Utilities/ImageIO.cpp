#include "ImageIO.h"

bool ImageIO::getImageDimensions(const QString &path,
                                 ImageDimensions &dimensions) {
  QImage qImage(path);
  if (qImage.isNull()) {
    return false;
  }
  dimensions.rows = qImage.height();
  dimensions.cols = qImage.width();
  return true;
}

/**
 * @brief floatToChar Converts a float in the range [0, 1] to a char.
 * @param value the float value
 * @return the char value [0, 255]
 */
int ImageIO::floatToChar(float value) {
  return std::min(255, int(value * 256));
}

/**
 * @brief charToFloat Converts an int in the range [0, 255] to a float in the
 * range [0, 1].
 * @param value the int value
 * @return the float value [0, 1]
 */
float ImageIO::charToFloat(int value) { return float(value) / 255.f; }

/**
 * @brief floatsToPixel Converts floating point RGBA values to an integer that
 * encodes each channel in one char (0-255). The format matchs qRgb, whose order
 * is ARGB.
 * @param r red
 * @param g green
 * @param b blue
 * @param a alpha
 * @return an integer that stores RGBA in qRgb's format (the order is ARGB)
 */
int ImageIO::floatsToPixel(float r, float g, float b, float a) {
  return (floatToChar(a) << 24) + (floatToChar(r) << 16) +
         (floatToChar(g) << 8) + floatToChar(b);
}

void ImageIO::pixelToFloats(int pixel, float &r, float &g, float &b, float &a) {
  r = charToFloat(qRed(pixel));
  g = charToFloat(qGreen(pixel));
  b = charToFloat(qBlue(pixel));
  a = charToFloat(qAlpha(pixel));
}
