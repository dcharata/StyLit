#ifndef UTILITIES_H_
#define UTILITIES_H_

namespace StyLitCUDA {

#define check(ans)                                                                                 \
  { assertCUDA((ans), __FILE__, __LINE__); }
void assertCUDA(cudaError_t code, const char *file, int line, bool abort = true);

} /* namespace StyLitCUDA */

#endif /* UTILITIES_H_ */
