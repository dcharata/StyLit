#ifndef IMAGEDIMENSIONS_H
#define IMAGEDIMENSIONS_H

/**
 * @brief The ImageDimensions struct This represents an image's dimensions.
 * This may seem unnecessary, but I think it helps to have everything in terms
 * of width/height or rows/columns.
 * Note: ImageCoordinates is identical to ImageDimensions. The unions allow you
 * to use either row/rows or col/cols to refer to the same thing depending on
 * whether the struct is thought of as an ImageCoordinates or an
 * ImageDimensions. I thought this might make some functions (e.g. checking
 * bounds) more clear. Feel free to change this if you find it confusing though.
 */
struct ImageDimensions {
  // the number of rows in the image (i.e. the image's height)
  union {
    int rows;
    int row;
  };

  // the number of columns in the image (i.e. the image's width)
  union {
    int cols;
    int col;
  };

  /**
   * @brief area Calculates the dimensions' area.
   * @return the dimensions' area
   */
  int area() const;

  /**
   * @brief includesCoordinates Checks whether these ImageCoordinates are within
   * the specified ImageDimensions.
   * @param dimensions the dimensions to check against
   * @return true if this ImageCoordinates is within the dimensions; otherwise
   * false
   */
  bool within(const ImageDimensions &dimensions) const;

  /**
   * @brief halfTheSizeOf Checks whether these ImageDimensions are half the
   * size of the specified imageDimensions in each dimension.
   * @param dimensions the dimensions to check against
   * @return true if so; otherwise false
   */
  bool halfTheSizeOf(const ImageDimensions &dimensions) const;
};

typedef ImageDimensions ImageCoordinates;

#endif // IMAGEDIMENSIONS_H
