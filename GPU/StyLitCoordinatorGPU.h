#ifndef STYLITCOORDINATORGPU_H
#define STYLITCOORDINATORGPU_H

#include "Algorithm/Pyramid.h"
#include "Algorithm/PyramidLevel.h"
#include "Algorithm/StyLitCoordinator.h"
#include "Utilities/ImageIO.h"

#include <QString>
#include <iostream>

template <unsigned int numGuideChannels, unsigned int numStyleChannels>
class StyLitCoordinatorGPU : public StyLitCoordinator<float, numGuideChannels, numStyleChannels> {
public:
  StyLitCoordinatorGPU() = default;
  ~StyLitCoordinatorGPU() = default;

  /**
   * @brief runStyLit The implementation-specific StyLit implementation is
   * called from here.
   * @return true if StyLit ran successfully; otherwise false
   */
  bool runStyLit(const Configuration &configuration) {
    startTimer();
    std::cout << std::endl << std::endl;
    printTime("Starting runStyLit in StyLitCoordinatorGPU.");

    // Gets the highest pyramid level's dimensions.
    ImageDimensions sourceDimensions;
    if (!ImageIO::getImageDimensions(configuration.sourceGuideImagePaths[0], sourceDimensions)) {
      std::cerr << "Could not read input images." << std::endl;
      return false;
    }
    ImageDimensions targetDimensions;
    if (!ImageIO::getImageDimensions(configuration.targetGuideImagePaths[0], targetDimensions)) {
      std::cerr << "Could not read input images." << std::endl;
      return false;
    }

    // Creates and reads in the highest pyramid level.
    PyramidLevel<float, numGuideChannels, numStyleChannels> pyramidLevel(sourceDimensions,
                                                                         targetDimensions);
    if (!ImageIO::readPyramidLevel<numGuideChannels, numStyleChannels>(configuration,
                                                                       pyramidLevel)) {
      std::cerr << "Could not read input images." << std::endl;
      return false;
    }
    printTime("Done reading A, B and A'.");

    // Adds the guide and style weights.
    unsigned int guideChannel = 0;
    for (unsigned int i = 0; i < configuration.guideImageFormats.size(); i++) {
      const int numChannels = ImageFormatTools::numChannels(configuration.guideImageFormats[i]);
      for (int j = 0; j < numChannels; j++) {
        // pyramid.guideWeights[guideChannel++] = configuration.guideImageWeights[i];
      }
    }
    Q_ASSERT(guideChannel == numGuideChannels);
    unsigned int styleChannel = 0;
    for (unsigned int i = 0; i < configuration.styleImageFormats.size(); i++) {
      const int numChannels = ImageFormatTools::numChannels(configuration.styleImageFormats[i]);
      for (int j = 0; j < numChannels; j++) {
        // pyramid.styleWeights[styleChannel++] = configuration.styleImageWeights[i];
      }
    }
    Q_ASSERT(styleChannel == numStyleChannels);

    return true;
  }

private:
  // This is used to print execution times to stdout.
  std::chrono::time_point<std::chrono::steady_clock> startTime;

  // Starts the timer.
  void startTimer() { startTime = std::chrono::steady_clock::now(); }

  /**
   * @brief printTime Prints the number of milliseconds since the start of
   * program execution along with the event.
   * @param event the event string
   */
  void printTime(const QString &event) {
    auto end = std::chrono::steady_clock::now();
    auto diff = end - startTime;
    std::cout << std::chrono::duration<double, std::milli>(diff).count()
              << " ms: " << event.toLocal8Bit().constData() << std::endl;
  }
};

#endif // STYLITCOORDINATORGPU_H
