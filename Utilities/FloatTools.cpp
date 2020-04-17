#include "FloatTools.h"

#include <cmath>

bool FloatTools::tolerantEquals(float lhs, float rhs) {
  return std::abs(lhs - rhs) <= TOLERANT_EQUALS_EPSILON;
}
