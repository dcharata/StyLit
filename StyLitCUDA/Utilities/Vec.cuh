#ifndef VEC_H_
#define VEC_H_

namespace StyLitCUDA {

// This is kind of reinventing the wheel, but for some reason including <thrust/device_vector.h>
// prevents anything from compiling on my computer, and I don't want to risk reinstalling Linux this
// close to the deadline.
template <typename T> struct Vec {
  Vec(int size);
  void deviceAllocate();
  void deviceFree();
  void toDevice(T *data);
  void fromDevice(T *data);
  int size;
  T *deviceData = nullptr;
};

} /* namespace StyLitCUDA */

#endif /* VEC_H_ */
