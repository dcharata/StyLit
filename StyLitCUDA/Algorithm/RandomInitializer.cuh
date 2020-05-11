#ifndef RANDOMINITIALIZER_H_
#define RANDOMINITIALIZER_H_

#include "../Utilities/Image.cuh"
#include "PCG.cuh"

namespace StyLitCUDA {

void initializeRandomState(const Image<PCGState> &random);

} /* namespace StyLitCUDA */

#endif /* RANDOMINITIALIZER_H_ */
