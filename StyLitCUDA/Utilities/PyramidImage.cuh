#ifndef READONLYIMAGE_H_
#define READONLYIMAGE_H_

namespace StyLitCUDA {

template <typename T> class PyramidImage {
public:
  PyramidImage(const int rows, const int cols, const int numChannels, const int numLevels);
  virtual ~PyramidImage();

  virtual void allocate() = 0;

  virtual void free() = 0;

  __device__ virtual const T *at(const int row, const int col, const int level) = 0;

protected:
  const int rows;
  const int cols;
  const int numChannels;
  const int numLevels;
};

} /* namespace StyLitCUDA */

#endif /* READONLYIMAGE_H_ */
