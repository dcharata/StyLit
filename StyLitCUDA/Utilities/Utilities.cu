#include "Utilities.cuh"

#include <stdio.h>

namespace StyLitCUDA {

void assertCUDA(cudaError_t code, const char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr, "epic CUDA fail: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

namespace Utilities {

int divideRoundUp(int a, int b) { return (a + b - 1) / b; }

__device__ int restrict(int value, int max) { return (value >= max) ? max - 1 : value; }

__device__ int clamp(const int minInclusive, const int value, const int maxExclusive) {
  if (value < minInclusive) {
    return minInclusive;
  }
  if (value >= maxExclusive) {
    return maxExclusive - 1;
  }
  return value;
}

} /* namespace Utilities */

} /* namespace StyLitCUDA */
