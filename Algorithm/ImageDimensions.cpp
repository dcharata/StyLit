#include "ImageDimensions.h"

const ImageDimensions ImageDimensions::FREE_PATCH = ImageDimensions{-1,-1};

ImageDimensions::ImageDimensions() : rows(-1), cols(-1) {}

ImageDimensions::ImageDimensions(int rows, int cols) : rows(rows), cols(cols) {}

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

ImageDimensions operator/(const ImageDimensions &dimensions, const int n) {
    return ImageDimensions {dimensions.row / n, dimensions.col / n};
}

ImageDimensions operator+(const ImageDimensions &dim1, const ImageDimensions &dim2) {
    return ImageDimensions {dim1.row + dim2.row, dim1.col + dim2.col};
}
