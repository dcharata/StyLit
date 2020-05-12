#ifndef REVERSETOFORWARDNNF_H_
#define REVERSETOFORWARDNNF_H_

#include "../Utilities/Image.cuh"
#include "NNFEntry.h"

namespace StyLitCUDA {
namespace ReverseToForwardNNF {

/**
 * @brief transfer Transfers mappings from the reverse NNF to the forward NNF where the forward NNF
 * does not yet have mappings. Only transfers mappings whose errors are below a knee point.
 * @param reverse the reverse NNF
 * @param forward the forward NNF
 * @return the number of patches transferred
 */
int transfer(Image<NNFEntry> &reverse, Image<NNFEntry> &forward);

/**
 * @brief fill Fills the NNF to, which is only partially filled (some mappings are to NNF::INVALID),
 * with the corresponding mappings from the NNF from.
 * @param from a fully filled NNF
 * @param to a partially filled NNF
 */
void fill(Image<NNFEntry> &from, Image<NNFEntry> &to);

} /* namespace ReverseToForwardNNF */
} /* namespace StyLitCUDA */

#endif /* REVERSETOFORWARDNNF_H_ */
