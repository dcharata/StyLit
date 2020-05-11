#ifndef COORDINATES_H_
#define COORDINATES_H_

namespace StyLitCUDA {

struct Coordinates {
  __device__ Coordinates();
  __device__ Coordinates(const int row, const int col);
  int row;
  int col;
};

__device__ Coordinates operator/(const Coordinates &lhs, const int rhs);

} /* namespace StyLitCUDA */

#endif /* COORDINATES_H_ */
