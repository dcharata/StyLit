#ifndef IMAGE_H
#define IMAGE_H

#include "FeatureVector.h"
#include "ImageDimensions.h"

#include <memory>

/**
 * @brief The Image struct This defines an image of feature vectors. When
 * possible, try to access individual feature vectors using operator ().
 */
template <typename T, unsigned int numChannels> struct Image {
public:
  /**
   * @brief Image Creates an image of the specified size.
   * @param dimensions the image's dimensions
   */
  Image(ImageDimensions dimensions) : dimensions(dimensions) {
    data = std::make_unique<FeatureVector<T, numChannels>[]>(dimensions.area());
  }

  /**
   * @brief operator () Returns a reference to the feature vector at the given
   * coordinates.
   * @param row the row
   * @param col the column
   * @return a reference to the feature vector at the given coordinates
   */
  FeatureVector<T, numChannels> &operator()(int row, int col) const {
    return data[dimensions.cols * row + col];
  }

  // the image's dimensions
  const ImageDimensions dimensions;

  // For now, this is a unique_ptr to an array. It could arguably be something
  // else, like a std::vector. Feel free to change this if you have a good
  // reason to think something else would be better.
  std::unique_ptr<FeatureVector<T, numChannels>[]> data;
};

#endif // IMAGE_H
