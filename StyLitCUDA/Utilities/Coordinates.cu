#include "Coordinates.cuh"

namespace StyLitCUDA {

__device__ Coordinates::Coordinates() : row(0), col(0) {}

__device__ Coordinates::Coordinates(const int row, const int col) : row(row), col(col) {}

__device__ Coordinates operator/(const Coordinates &lhs, const int rhs) {
  return Coordinates(lhs.row / rhs, lhs.col / rhs);
}

} /* namespace StyLitCUDA */
