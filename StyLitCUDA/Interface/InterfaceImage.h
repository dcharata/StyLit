#ifndef INTERFACEIMAGE_H_
#define INTERFACEIMAGE_H_

namespace StyLitCUDA {

template <typename T> struct InterfaceImage {
  /**
   * @brief at Returns a pointer to the feature vector for the given coordinates.
   * @param row the row
   * @param col the column
   * @return a pointer to the feature vector for the given coordinates
   */
  T *at(const int row, const int col) {
    const int index = row * cols + col;
    return &data[index * numChannels];
  }

  /**
   * @brief constAt Returns a const pointer to the feature vector for the given coordinates.
   * @param row the row
   * @param col the column
   * @return a const pointer to the feature vector for the given coordinates
   */
  const T *constAt(const int row, const int col) const {
    const int index = row * cols + col;
    return &data[index * numChannels];
  }

  // the number of rows in the image (i.e. its height)
  int rows;

  // the number of columns in the image (i.e. its width)
  int cols;

  // The number of channels in the image. The number of bytes allocated to each pixel is channels *
  // sizeof(T).
  int numChannels;

  // A pointer to the image's data on the host device. This should be row-major.
  T *data;
};

} /* namespace StyLitCUDA */

#endif /* INTERFACEIMAGE_H_ */
