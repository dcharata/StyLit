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

  // the patch size (a patch consists of patchSize * patchSize pixels)
  int patchSize = 0;

  // the number of PatchMatch iterations
  int numPatchMatchIterations = 0;

  // the number of pyramid levels used for StyLit
  int numPyramidLevels = 0;
};

#endif // CONFIGURATION_H
