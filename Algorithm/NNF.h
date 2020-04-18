#ifndef NNF_H
#define NNF_H

#include "FeatureVector.h"
#include "ImageDimensions.h"

#include <memory>

/**
 * @brief The NNF class This represents a NNF (nearest-neighbor field).
 * Note that the source and destination don't necessarily have to be B/B' and
 * A/B' respectively. The terminology we've been using is the following:
 * NNF: B-sized 2D array of indices in A.
 * Reverse NNF: A-sized 2D array of indices in B.
 * This is a class and not a struct so that getMappings() has to be used when
 * accessing the mappings directly instead of through the safer getter and
 * setter.
 */
class NNF {
public:
  NNF(ImageDimensions sourceDimensions, ImageDimensions targetDimensions);

  /**
   * @brief getMappings Returns a reference to the mappings. If possible, try to
   * use the get and set methods instead, since they'll assert correct bounds.
   * @return the NNF's mappings
   */
  std::unique_ptr<ImageCoordinates[]> &getMappings();

  /**
   * @brief getMapping Gets an NNF mapping.
   * @param from the coordinates that are mapped from
   * @return the coordinates that are mapped to
   */
  ImageCoordinates getMapping(const ImageCoordinates &from) const;

  /**
   * @brief setMapping Sets a mapping in the NNF.
   * @param from the coordinates in the source to map from
   * @param to the coordinates in the target to map to
   */
  void setMapping(const ImageCoordinates &from, const ImageCoordinates &to);

  void setToInitializedBlacklist();

  // the source image's size
  const ImageDimensions sourceDimensions;

  // the target image's size
  const ImageDimensions targetDimensions;

private:
  // this is where the NNF's mappings are stored
  std::unique_ptr<ImageCoordinates[]> mappings;
};

#endif // NNF_H
