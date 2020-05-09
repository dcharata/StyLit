#include "ImplementationSelector.h"

#include "CPU/StyLitCoordinatorCPU.h"
#include "GPU/StyLitCoordinatorGPU.h"
#include "Utilities/ImageFormat.h"
#include "Utilities/ImageFormatTools.h"

bool ImplementationSelector::runWithConfiguration(const Configuration &configuration) {
  // Counts the number of style and guide channels.
  int numGuideChannels = 0;
  int numStyleChannels = 0;
  for (const ImageFormat &guideFormat : configuration.guideImageFormats) {
    numGuideChannels += ImageFormatTools::numChannels(guideFormat);
  }
  for (const ImageFormat &styleFormat : configuration.styleImageFormats) {
    numStyleChannels += ImageFormatTools::numChannels(styleFormat);
  }

  // Picks the right implementation.
  if (configuration.coordinatorType == Configuration::CoordinatorType::CPU) {
    return runCPU(configuration);
  } else {
    return runCUDA(configuration);
  }
}

bool ImplementationSelector::runCPU(const Configuration &configuration) {
  // There can be up to 4 style channels and 20 guide channels.
  if (configuration.numGuideChannels > 20 || configuration.numStyleChannels > 4) {
    std::cerr << "The CPU implementation supports up to 20 guide channels and "
                 "4 style channels.";
    return false;
  }

  StyLitCoordinatorCPU<1, 1> coord_1_1;
  StyLitCoordinatorCPU<1, 2> coord_1_2;
  StyLitCoordinatorCPU<1, 3> coord_1_3;
  StyLitCoordinatorCPU<1, 4> coord_1_4;

  StyLitCoordinatorCPU<2, 1> coord_2_1;
  StyLitCoordinatorCPU<2, 2> coord_2_2;
  StyLitCoordinatorCPU<2, 3> coord_2_3;
  StyLitCoordinatorCPU<2, 4> coord_2_4;

  StyLitCoordinatorCPU<3, 1> coord_3_1;
  StyLitCoordinatorCPU<3, 2> coord_3_2;
  StyLitCoordinatorCPU<3, 3> coord_3_3;
  StyLitCoordinatorCPU<3, 4> coord_3_4;

  StyLitCoordinatorCPU<4, 1> coord_4_1;
  StyLitCoordinatorCPU<4, 2> coord_4_2;
  StyLitCoordinatorCPU<4, 3> coord_4_3;
  StyLitCoordinatorCPU<4, 4> coord_4_4;

  StyLitCoordinatorCPU<5, 1> coord_5_1;
  StyLitCoordinatorCPU<5, 2> coord_5_2;
  StyLitCoordinatorCPU<5, 3> coord_5_3;
  StyLitCoordinatorCPU<5, 4> coord_5_4;

  StyLitCoordinatorCPU<6, 1> coord_6_1;
  StyLitCoordinatorCPU<6, 2> coord_6_2;
  StyLitCoordinatorCPU<6, 3> coord_6_3;
  StyLitCoordinatorCPU<6, 4> coord_6_4;

  StyLitCoordinatorCPU<7, 1> coord_7_1;
  StyLitCoordinatorCPU<7, 2> coord_7_2;
  StyLitCoordinatorCPU<7, 3> coord_7_3;
  StyLitCoordinatorCPU<7, 4> coord_7_4;

  StyLitCoordinatorCPU<8, 1> coord_8_1;
  StyLitCoordinatorCPU<8, 2> coord_8_2;
  StyLitCoordinatorCPU<8, 3> coord_8_3;
  StyLitCoordinatorCPU<8, 4> coord_8_4;

  StyLitCoordinatorCPU<9, 1> coord_9_1;
  StyLitCoordinatorCPU<9, 2> coord_9_2;
  StyLitCoordinatorCPU<9, 3> coord_9_3;
  StyLitCoordinatorCPU<9, 4> coord_9_4;

  StyLitCoordinatorCPU<10, 1> coord_10_1;
  StyLitCoordinatorCPU<10, 2> coord_10_2;
  StyLitCoordinatorCPU<10, 3> coord_10_3;
  StyLitCoordinatorCPU<10, 4> coord_10_4;

  StyLitCoordinatorCPU<11, 1> coord_11_1;
  StyLitCoordinatorCPU<11, 2> coord_11_2;
  StyLitCoordinatorCPU<11, 3> coord_11_3;
  StyLitCoordinatorCPU<11, 4> coord_11_4;

  StyLitCoordinatorCPU<12, 1> coord_12_1;
  StyLitCoordinatorCPU<12, 2> coord_12_2;
  StyLitCoordinatorCPU<12, 3> coord_12_3;
  StyLitCoordinatorCPU<12, 4> coord_12_4;

  StyLitCoordinatorCPU<13, 1> coord_13_1;
  StyLitCoordinatorCPU<13, 2> coord_13_2;
  StyLitCoordinatorCPU<13, 3> coord_13_3;
  StyLitCoordinatorCPU<13, 4> coord_13_4;

  StyLitCoordinatorCPU<14, 1> coord_14_1;
  StyLitCoordinatorCPU<14, 2> coord_14_2;
  StyLitCoordinatorCPU<14, 3> coord_14_3;
  StyLitCoordinatorCPU<14, 4> coord_14_4;

  StyLitCoordinatorCPU<15, 1> coord_15_1;
  StyLitCoordinatorCPU<15, 2> coord_15_2;
  StyLitCoordinatorCPU<15, 3> coord_15_3;
  StyLitCoordinatorCPU<15, 4> coord_15_4;

  StyLitCoordinatorCPU<16, 1> coord_16_1;
  StyLitCoordinatorCPU<16, 2> coord_16_2;
  StyLitCoordinatorCPU<16, 3> coord_16_3;
  StyLitCoordinatorCPU<16, 4> coord_16_4;

  StyLitCoordinatorCPU<17, 1> coord_17_1;
  StyLitCoordinatorCPU<17, 2> coord_17_2;
  StyLitCoordinatorCPU<17, 3> coord_17_3;
  StyLitCoordinatorCPU<17, 4> coord_17_4;

  StyLitCoordinatorCPU<18, 1> coord_18_1;
  StyLitCoordinatorCPU<18, 2> coord_18_2;
  StyLitCoordinatorCPU<18, 3> coord_18_3;
  StyLitCoordinatorCPU<18, 4> coord_18_4;

  StyLitCoordinatorCPU<19, 1> coord_19_1;
  StyLitCoordinatorCPU<19, 2> coord_19_2;
  StyLitCoordinatorCPU<19, 3> coord_19_3;
  StyLitCoordinatorCPU<19, 4> coord_19_4;

  StyLitCoordinatorCPU<20, 1> coord_20_1;
  StyLitCoordinatorCPU<20, 2> coord_20_2;
  StyLitCoordinatorCPU<20, 3> coord_20_3;
  StyLitCoordinatorCPU<20, 4> coord_20_4;

  StyLitCoordinatorBase *implementations[20][4] = {
      {&coord_1_1, &coord_1_2, &coord_1_3, &coord_1_4},
      {&coord_2_1, &coord_2_2, &coord_2_3, &coord_2_4},
      {&coord_3_1, &coord_3_2, &coord_3_3, &coord_3_4},
      {&coord_4_1, &coord_4_2, &coord_4_3, &coord_4_4},
      {&coord_5_1, &coord_5_2, &coord_5_3, &coord_5_4},
      {&coord_6_1, &coord_6_2, &coord_6_3, &coord_6_4},
      {&coord_7_1, &coord_7_2, &coord_7_3, &coord_7_4},
      {&coord_8_1, &coord_8_2, &coord_8_3, &coord_8_4},
      {&coord_9_1, &coord_9_2, &coord_9_3, &coord_9_4},
      {&coord_10_1, &coord_10_2, &coord_10_3, &coord_10_4},
      {&coord_11_1, &coord_11_2, &coord_11_3, &coord_11_4},
      {&coord_12_1, &coord_12_2, &coord_12_3, &coord_12_4},
      {&coord_13_1, &coord_13_2, &coord_13_3, &coord_13_4},
      {&coord_14_1, &coord_14_2, &coord_14_3, &coord_14_4},
      {&coord_15_1, &coord_15_2, &coord_15_3, &coord_15_4},
      {&coord_16_1, &coord_16_2, &coord_16_3, &coord_16_4},
      {&coord_17_1, &coord_17_2, &coord_17_3, &coord_17_4},
      {&coord_18_1, &coord_18_2, &coord_18_3, &coord_18_4},
      {&coord_19_1, &coord_19_2, &coord_19_3, &coord_19_4},
      {&coord_20_1, &coord_20_2, &coord_20_3, &coord_20_4}};

  return implementations[configuration.numGuideChannels - 1][configuration.numStyleChannels - 1]
      ->runStyLit(configuration);
}

bool ImplementationSelector::runCUDA(const Configuration &configuration) {
  // There can be up to 4 style channels and 20 guide channels.
  if (configuration.numGuideChannels > 20 || configuration.numStyleChannels > 4) {
    std::cerr << "The CUDA implementation supports up to 20 guide channels and "
                 "4 style channels.";
    return false;
  }

  StyLitCoordinatorGPU<1, 1> coord_1_1;
  StyLitCoordinatorGPU<1, 2> coord_1_2;
  StyLitCoordinatorGPU<1, 3> coord_1_3;
  StyLitCoordinatorGPU<1, 4> coord_1_4;

  StyLitCoordinatorGPU<2, 1> coord_2_1;
  StyLitCoordinatorGPU<2, 2> coord_2_2;
  StyLitCoordinatorGPU<2, 3> coord_2_3;
  StyLitCoordinatorGPU<2, 4> coord_2_4;

  StyLitCoordinatorGPU<3, 1> coord_3_1;
  StyLitCoordinatorGPU<3, 2> coord_3_2;
  StyLitCoordinatorGPU<3, 3> coord_3_3;
  StyLitCoordinatorGPU<3, 4> coord_3_4;

  StyLitCoordinatorGPU<4, 1> coord_4_1;
  StyLitCoordinatorGPU<4, 2> coord_4_2;
  StyLitCoordinatorGPU<4, 3> coord_4_3;
  StyLitCoordinatorGPU<4, 4> coord_4_4;

  StyLitCoordinatorGPU<5, 1> coord_5_1;
  StyLitCoordinatorGPU<5, 2> coord_5_2;
  StyLitCoordinatorGPU<5, 3> coord_5_3;
  StyLitCoordinatorGPU<5, 4> coord_5_4;

  StyLitCoordinatorGPU<6, 1> coord_6_1;
  StyLitCoordinatorGPU<6, 2> coord_6_2;
  StyLitCoordinatorGPU<6, 3> coord_6_3;
  StyLitCoordinatorGPU<6, 4> coord_6_4;

  StyLitCoordinatorGPU<7, 1> coord_7_1;
  StyLitCoordinatorGPU<7, 2> coord_7_2;
  StyLitCoordinatorGPU<7, 3> coord_7_3;
  StyLitCoordinatorGPU<7, 4> coord_7_4;

  StyLitCoordinatorGPU<8, 1> coord_8_1;
  StyLitCoordinatorGPU<8, 2> coord_8_2;
  StyLitCoordinatorGPU<8, 3> coord_8_3;
  StyLitCoordinatorGPU<8, 4> coord_8_4;

  StyLitCoordinatorGPU<9, 1> coord_9_1;
  StyLitCoordinatorGPU<9, 2> coord_9_2;
  StyLitCoordinatorGPU<9, 3> coord_9_3;
  StyLitCoordinatorGPU<9, 4> coord_9_4;

  StyLitCoordinatorGPU<10, 1> coord_10_1;
  StyLitCoordinatorGPU<10, 2> coord_10_2;
  StyLitCoordinatorGPU<10, 3> coord_10_3;
  StyLitCoordinatorGPU<10, 4> coord_10_4;

  StyLitCoordinatorGPU<11, 1> coord_11_1;
  StyLitCoordinatorGPU<11, 2> coord_11_2;
  StyLitCoordinatorGPU<11, 3> coord_11_3;
  StyLitCoordinatorGPU<11, 4> coord_11_4;

  StyLitCoordinatorGPU<12, 1> coord_12_1;
  StyLitCoordinatorGPU<12, 2> coord_12_2;
  StyLitCoordinatorGPU<12, 3> coord_12_3;
  StyLitCoordinatorGPU<12, 4> coord_12_4;

  StyLitCoordinatorGPU<13, 1> coord_13_1;
  StyLitCoordinatorGPU<13, 2> coord_13_2;
  StyLitCoordinatorGPU<13, 3> coord_13_3;
  StyLitCoordinatorGPU<13, 4> coord_13_4;

  StyLitCoordinatorGPU<14, 1> coord_14_1;
  StyLitCoordinatorGPU<14, 2> coord_14_2;
  StyLitCoordinatorGPU<14, 3> coord_14_3;
  StyLitCoordinatorGPU<14, 4> coord_14_4;

  StyLitCoordinatorGPU<15, 1> coord_15_1;
  StyLitCoordinatorGPU<15, 2> coord_15_2;
  StyLitCoordinatorGPU<15, 3> coord_15_3;
  StyLitCoordinatorGPU<15, 4> coord_15_4;

  StyLitCoordinatorGPU<16, 1> coord_16_1;
  StyLitCoordinatorGPU<16, 2> coord_16_2;
  StyLitCoordinatorGPU<16, 3> coord_16_3;
  StyLitCoordinatorGPU<16, 4> coord_16_4;

  StyLitCoordinatorGPU<17, 1> coord_17_1;
  StyLitCoordinatorGPU<17, 2> coord_17_2;
  StyLitCoordinatorGPU<17, 3> coord_17_3;
  StyLitCoordinatorGPU<17, 4> coord_17_4;

  StyLitCoordinatorGPU<18, 1> coord_18_1;
  StyLitCoordinatorGPU<18, 2> coord_18_2;
  StyLitCoordinatorGPU<18, 3> coord_18_3;
  StyLitCoordinatorGPU<18, 4> coord_18_4;

  StyLitCoordinatorGPU<19, 1> coord_19_1;
  StyLitCoordinatorGPU<19, 2> coord_19_2;
  StyLitCoordinatorGPU<19, 3> coord_19_3;
  StyLitCoordinatorGPU<19, 4> coord_19_4;

  StyLitCoordinatorGPU<20, 1> coord_20_1;
  StyLitCoordinatorGPU<20, 2> coord_20_2;
  StyLitCoordinatorGPU<20, 3> coord_20_3;
  StyLitCoordinatorGPU<20, 4> coord_20_4;

  StyLitCoordinatorBase *implementations[20][4] = {
      {&coord_1_1, &coord_1_2, &coord_1_3, &coord_1_4},
      {&coord_2_1, &coord_2_2, &coord_2_3, &coord_2_4},
      {&coord_3_1, &coord_3_2, &coord_3_3, &coord_3_4},
      {&coord_4_1, &coord_4_2, &coord_4_3, &coord_4_4},
      {&coord_5_1, &coord_5_2, &coord_5_3, &coord_5_4},
      {&coord_6_1, &coord_6_2, &coord_6_3, &coord_6_4},
      {&coord_7_1, &coord_7_2, &coord_7_3, &coord_7_4},
      {&coord_8_1, &coord_8_2, &coord_8_3, &coord_8_4},
      {&coord_9_1, &coord_9_2, &coord_9_3, &coord_9_4},
      {&coord_10_1, &coord_10_2, &coord_10_3, &coord_10_4},
      {&coord_11_1, &coord_11_2, &coord_11_3, &coord_11_4},
      {&coord_12_1, &coord_12_2, &coord_12_3, &coord_12_4},
      {&coord_13_1, &coord_13_2, &coord_13_3, &coord_13_4},
      {&coord_14_1, &coord_14_2, &coord_14_3, &coord_14_4},
      {&coord_15_1, &coord_15_2, &coord_15_3, &coord_15_4},
      {&coord_16_1, &coord_16_2, &coord_16_3, &coord_16_4},
      {&coord_17_1, &coord_17_2, &coord_17_3, &coord_17_4},
      {&coord_18_1, &coord_18_2, &coord_18_3, &coord_18_4},
      {&coord_19_1, &coord_19_2, &coord_19_3, &coord_19_4},
      {&coord_20_1, &coord_20_2, &coord_20_3, &coord_20_4}};

  return implementations[configuration.numGuideChannels - 1][configuration.numStyleChannels - 1]
      ->runStyLit(configuration);
}
