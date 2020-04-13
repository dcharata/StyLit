#ifndef NNFERROR_H
#define NNFERROR_H

#include "Image.h"

class NNF;

/**
 * @brief The NNFError struct This struct holds a float error for each source
 * position in the supplied NNF. It's given to the knee point finding algorithm.
 */
struct NNFError {
  NNFError(const NNF &nnf);

  // the NNF whose error is stored
  const NNF &nnf;

  // where the errors reside
  Image<float, 1> error;
};

#endif // NNFERROR_H
