#ifndef DOWNSCALER_H_
#define DOWNSCALER_H_

#include "../Utilities/Image.cuh"

namespace StyLitCUDA {

template <typename T> void downscale(const Image<T> &from, Image<T> &to);

} /* namespace StyLitCUDA */

#endif /* DOWNSCALER_H_ */
