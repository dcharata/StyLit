#include "Downscaler.cuh"

#include <stdio.h>

namespace StyLitCUDA {
namespace Downscaler {



template <typename T>
void downscale(const ImagePitch<T> &from, ImagePitch<T> to) {
  printf("StyLitCUDA: Downscaling [%d, %d] to [%d, %d].\n", from.rows, from.cols, to.rows, to.cols);
}

template void downscale(const ImagePitch<int> &from, ImagePitch<int> to);
template void downscale(const ImagePitch<float> &from, ImagePitch<float> to);

} /* namespace Downscaler */
} /* namespace StyLitCUDA */
