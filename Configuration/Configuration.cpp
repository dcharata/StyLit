#include "Configuration.h"

#include <iostream>
#include <stdio.h>

#include "Utilities/ImageFormatTools.h"

using namespace std;

void Configuration::print() {
  cout << "Configuration:" << endl << endl;

  cout << "Source guide image paths:" << endl;
  for (const QString &path : sourceGuideImagePaths) {
    cout << path.toLocal8Bit().constData() << endl;
  }
  cout << endl;

  cout << "Target guide image paths:" << endl;
  for (const QString &path : targetGuideImagePaths) {
    cout << path.toLocal8Bit().constData() << endl;
  }
  cout << endl;

  cout << "Source style image paths:" << endl;
  for (const QString &path : sourceStyleImagePaths) {
    cout << path.toLocal8Bit().constData() << endl;
  }
  cout << endl;

  cout << "Target style image paths:" << endl;
  for (const QString &path : targetStyleImagePaths) {
    cout << path.toLocal8Bit().constData() << endl;
  }
  cout << endl;

  cout << "Guide image formats:" << endl;
  for (unsigned int i = 0; i < guideImageFormats.size(); i++) {
    cout << ImageFormatTools::imageFormatToString(guideImageFormats[i])
         << " (weight: " << guideImageWeights[i] << ")" << endl;
  }
  cout << endl;

  cout << "Style image formats:" << endl;
  for (unsigned int i = 0; i < styleImageFormats.size(); i++) {
    cout << ImageFormatTools::imageFormatToString(styleImageFormats[i])
         << " (weight: " << styleImageWeights[i] << ")" << endl;
  }
  cout << endl;

  printf("Patch size: %d\n", patchSize);
  printf("Number of PatchMatch iterations: %d\n", numPatchMatchIterations);
  printf("Number of pyramid levels: %d\n", numPyramidLevels);
  printf("Number of optimization iterations per pyramid level: %d\n",
         numOptimizationIterationsPerPyramidLevel);
}
