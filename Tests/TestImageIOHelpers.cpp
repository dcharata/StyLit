#include "TestImageIOHelpers.h"

#include "Utilities/FloatTools.h"
#include "Utilities/ImageIO.h"

#include <cmath>
#include <iostream>

bool TestImageIOHelpers::run() {
  TEST_ASSERT(ImageIO::floatToChar(0.f) == 0);
  TEST_ASSERT(ImageIO::floatToChar(1.f) == 255);
  TEST_ASSERT(ImageIO::floatToChar(0.5f) == 128);

  TEST_ASSERT(ImageIO::charToFloat(0) == 0.f);
  TEST_ASSERT(ImageIO::charToFloat(255) == 1.f);

  // Going from char to float and back should definitely produce the same thing.
  for (int i = 0; i < 256; i++) {
    TEST_ASSERT(ImageIO::floatToChar(ImageIO::charToFloat(i)) == i);
  }

  // When going from float to char and back, correctness is a bit looser.
  for (float i = 0.f; i < 1.f; i += 0.001f) {
    TEST_ASSERT(std::abs(ImageIO::charToFloat(ImageIO::floatToChar(i)) - i) <
                1.f / 256.f);
  }

  TEST_ASSERT(ImageIO::floatsToPixel(1.f, 1.f, 1.f, 1.f) == int(0xFFFFFFFF));
  TEST_ASSERT(ImageIO::floatsToPixel(1.f, 0.f, 0.f, 0.f) == int(0x00FF0000));
  TEST_ASSERT(ImageIO::floatsToPixel(0.f, 1.f, 0.f, 0.f) == int(0x0000FF00));
  TEST_ASSERT(ImageIO::floatsToPixel(0.f, 0.f, 1.f, 0.f) == int(0x000000FF));
  TEST_ASSERT(ImageIO::floatsToPixel(0.f, 0.f, 0.f, 1.f) == int(0xFF000000));
  TEST_ASSERT(ImageIO::floatsToPixel(0.f, 0.f, 0.f, 0.f) == int(0x00000000));

  float r, g, b, a;
  ImageIO::pixelToFloats(0xFFFFFFFF, r, g, b, a);
  TEST_ASSERT(r == 1.f && g == 1.f && b == 1.f && a == 1.f);
  ImageIO::pixelToFloats(0x00000000, r, g, b, a);
  TEST_ASSERT(r == 0.f && g == 0.f && b == 0.f && a == 0.f);
  ImageIO::pixelToFloats(0xFF000000, r, g, b, a);
  TEST_ASSERT(r == 0.f && g == 0.f && b == 0.f && a == 1.f);
  ImageIO::pixelToFloats(0x00FF0000, r, g, b, a);
  TEST_ASSERT(r == 1.f && g == 0.f && b == 0.f && a == 0.f);
  ImageIO::pixelToFloats(0x0000FF00, r, g, b, a);
  TEST_ASSERT(r == 0.f && g == 1.f && b == 0.f && a == 0.f);
  ImageIO::pixelToFloats(0x000000FF, r, g, b, a);
  TEST_ASSERT(r == 0.f && g == 0.f && b == 1.f && a == 0.f);

  return true;
}
