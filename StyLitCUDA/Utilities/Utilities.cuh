#ifndef UTILITIES_H_
#define UTILITIES_H_

namespace StyLitCUDA {

#define check(ans)                                                                                 \
  { assertCUDA((ans), __FILE__, __LINE__); }
void assertCUDA(cudaError_t code, const char *file, int line, bool abort = true);

namespace Utilities {

/**
 * @brief divideRoundUp Does integer division, but rounds up instead of down. Only works on positive
 * numbers.
 * @param a the dividend
 * @param b the divisor
 * @return a divided by b, except it rounds up
 */
int divideRoundUp(int a, int b);

/**
 * @brief restrict Restricts the value to [0, max).
 * @param value the image to sample
 * @param max the row to sample at
 * @return [0, max)
 */
__device__ int restrict(int value, int max);

} /* namespace Utilities */
} /* namespace StyLitCUDA */

#endif /* UTILITIES_H_ */
