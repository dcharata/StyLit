#ifndef DOWNSCALER_H_
#define DOWNSCALER_H_

#include "../Utilities/ImagePitch.cuh"

namespace StyLitCUDA {
namespace Downscaler {

template <typename T>
void downscale(const ImagePitch<T> &from, ImagePitch<T> to);

} /* namespace Downscaler */
} /* namespace StyLitCUDA */

#endif /* DOWNSCALER_H_ */
