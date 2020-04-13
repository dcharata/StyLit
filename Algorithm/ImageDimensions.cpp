#include "ImageDimensions.h"

int ImageDimensions::area() const { return rows * cols; }

bool ImageDimensions::within(const ImageDimensions &dimensions) const {
  return row >= 0 && col >= 0 && row < dimensions.rows && col < dimensions.cols;
}

bool ImageDimensions::halfTheSizeOf(const ImageDimensions &dimensions) const {
  return rows == dimensions.rows / 2 && cols == dimensions.cols / 2;
}

bool operator==(const ImageDimensions &lhs, const ImageDimensions &rhs) {
  return lhs.rows == rhs.rows && lhs.cols == rhs.cols;
}

ImageDimensions operator*(const ImageDimensions &dimensions, const int n) {
    return ImageDimensions {dimensions.row * n, dimensions.col * n};
}
