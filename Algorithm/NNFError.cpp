#include "NNFError.h"

NNFError::NNFError(const NNF &nnf) : nnf(nnf), error(nnf.sourceDimensions) {}
