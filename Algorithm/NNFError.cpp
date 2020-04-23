#include "NNFError.h"

NNFError::NNFError(const NNF &nnf)
    : nnf(nnf), error(nnf.sourceDimensions), errorIndex(nnf.sourceDimensions) {}
