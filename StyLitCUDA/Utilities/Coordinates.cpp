#include "Coordinates.h"

namespace StyLitCUDA {

Coordinates::Coordinates() : row(0), col(0) {}

Coordinates::Coordinates(const int row, const int col) : row(row), col(col) {}

Coordinates operator/(const Coordinates &lhs, const int rhs) {
  return Coordinates(lhs.row / rhs, lhs.col / rhs);
}

} /* namespace StyLitCUDA */
