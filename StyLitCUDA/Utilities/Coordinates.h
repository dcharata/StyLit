#ifndef COORDINATES_H_
#define COORDINATES_H_

namespace StyLitCUDA {

struct Coordinates {
  Coordinates();
  Coordinates(const int row, const int col);
  virtual ~Coordinates() = default;
  int row;
  int col;
};

Coordinates operator/(const Coordinates &lhs, const int rhs);

} /* namespace StyLitCUDA */

#endif /* COORDINATES_H_ */
