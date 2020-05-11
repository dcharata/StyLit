#ifndef DOWNSCALER_H_
#define DOWNSCALER_H_

#include "../Utilities/Image.cuh"

namespace StyLitCUDA {
namespace Downscaler {

template <typename T> void downscale(const Image<T> &from, Image<T> to);

} /* namespace Downscaler */
} /* namespace StyLitCUDA */

#endif /* DOWNSCALER_H_ */
