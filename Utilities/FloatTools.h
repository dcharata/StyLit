#ifndef FLOATTOOLS_H
#define FLOATTOOLS_H

namespace FloatTools {
const float TOLERANT_EQUALS_EPSILON = 1e-5f;

/**
 * @brief tolerantEquals Returns true if lhs and rhs are within
 * TOLERANT_EQUALS_EPSILON of each other.
 * @param lhs left-hand side of equality check
 * @param rhs right-hand side of equality check
 * @return true if lhs and rhs are within TOLERANT_EQUALS_EPSILON of each other;
 * otherwise false
 */
bool tolerantEquals(float lhs, float rhs);
}; // namespace FloatTools

#endif // FLOATTOOLS_H
