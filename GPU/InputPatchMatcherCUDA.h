#ifndef INPUTPATCHMATCHERCUDA_H
#define INPUTPATCHMATCHERCUDA_H

/**
 * @brief The InputPatchMatcherCUDA struct This is the input to PatchMatcherCUDA. It's defined as a
 * struct to avoid making PatchMatcherCUDA have a ridiculous number of arguments.
 */
template <typename T> struct InputPatchMatcherCUDA {
  int sourceRows;
  int sourceCols;
  int targetRows;
  int targetCols;

  int numGuideChannels;
  int numStyleChannels;

  // The NNF's dimensions should be the same as the source's.
  // TODO: The error isn't currently written back to the host, but it should be.
  int *hostNNF;

  // If a blacklist is specified, the blacklists's dimensions should be the same as the target's.
  const void *hostBlacklist;

  const T *hostGuideSource;
  const T *hostGuideTarget;
  const T *hostStyleSource;
  const T *hostStyleTarget;
};

#endif // INPUTPATCHMATCHERCUDA_H
