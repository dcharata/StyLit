#include "ImageFormatTools.h"

using namespace std;

string ImageFormatTools::imageFormatToString(const ImageFormat &imageFormat) {
  switch (imageFormat) {
  case ImageFormat::BW:
    return "BW (black and white, 1 channel)";
  case ImageFormat::BWA:
    return "BWA (black and white plus alpha, 3 channels)";
  case ImageFormat::RGB:
    return "RGB (red, green and blue, 3 channels)";
  case ImageFormat::RGBA:
    return "RGBA (red, green and blue plus alpha, 4 channels)";
  default:
    return "Unknown image format.";
  }
}

int ImageFormatTools::numChannels(const ImageFormat &imageFormat) {
  switch (imageFormat) {
  case ImageFormat::BW:
    return 1;
  case ImageFormat::BWA:
    return 2;
  case ImageFormat::RGB:
    return 3;
  case ImageFormat::RGBA:
    return 4;
  default:
    return -1;
  }
}
