#ifndef INTERFACEIMAGE_H_
#define INTERFACEIMAGE_H_

template<typename T> struct InterfaceImage {
public:
  InterfaceImage(const int rows, const int cols, const int channels, const T * const data);
  virtual ~InterfaceImage() = default;

private:
  const int rows;
  const int cols;
  const int channels;
  const T * const data;
};

#endif /* INTERFACEIMAGE_H_ */
