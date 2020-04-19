#ifndef IMAGEFORMAT_H
#define IMAGEFORMAT_H

enum class ImageFormat {
  // RGB image (3 channels)
  RGB,

  // RGB image plus alpha (4 channels)
  RGBA,

  // black and white image (1 channel)
  BW,

  // black and white image plus alpha (2 channels)
  BWA
};

#endif // IMAGEFORMAT_H
