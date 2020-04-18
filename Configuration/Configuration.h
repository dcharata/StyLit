#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <QString>
#include <vector>

#include "Utilities/ImageFormat.h"

struct Configuration {
  Configuration() = default;

  // Prints the configuration to stdout.
  void print();

  // input paths for the images in A
  std::vector<QString> sourceGuideImagePaths;

  // input paths for the images in B
  std::vector<QString> targetGuideImagePaths;

  // input paths for the images in A'
  std::vector<QString> sourceStyleImagePaths;

  // output paths for the images in B'
  std::vector<QString> targetStyleImagePaths;

  // how guide images are interpreted (see ImageFormat.h)
  std::vector<ImageFormat> guideImageFormats;

  // how style images are interpreted (see ImageFormat.h)
  std::vector<ImageFormat> styleImageFormats;
};

#endif // CONFIGURATION_H
