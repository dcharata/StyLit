#include "ImageDimensions.h"

int ImageDimensions::area() const { return rows * cols; }

bool ImageDimensions::within(const ImageDimensions &dimensions) const {
  return row >= 0 && col >= 0 && row < dimensions.rows && col < dimensions.cols;
}

bool ImageDimensions::halfTheSizeOf(const ImageDimensions &dimensions) const {
  return rows == dimensions.rows / 2 && cols == dimensions.cols / 2;
}
