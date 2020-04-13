#include "PatchMatcher.h"

bool PatchMatcher::patchMatch(const Configuration &configuration, NNF &nnf,
                              const NNF *const blacklist) {
  return implementationOfPatchMatch(configuration, nnf, blacklist);
}
