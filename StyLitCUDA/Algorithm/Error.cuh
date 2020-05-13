#ifndef ERROR_H_
#define ERROR_H_

#include "../Utilities/Vec.cuh"
#include "../Utilities/Coordinates.cuh"
#include "../Utilities/Image.cuh"

namespace StyLitCUDA {
namespace Error {

template <typename T>
__device__ float calculate(const Image<T> &x, const Image<T> &y, Coordinates inX, Coordinates inY,
                           const int patchSize, const float *weights);

} /* namespace Error */
} /* namespace StyLitCUDA */

#endif /* ERROR_H_ */
