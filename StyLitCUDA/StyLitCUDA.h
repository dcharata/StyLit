#ifndef STYLITMAIN_H_
#define STYLITMAIN_H_

#include "Interface/InterfaceInput.h"

#include <stdio.h>

// Returns 0xDEADBEEF to indicate that the library linking is working as intended. This isn't in a
// namespace because this way, it's easy to forward declare it and avoid including headers in QT.
// There's probably a better way to do this, but it works for now.
unsigned int StyLitCUDA_sanityCheckStyLitCUDA();

int StyLitCUDA_runStyLitCUDA_float(StyLitCUDA::InterfaceInput<float> &input);

#endif /* STYLITMAIN_H_ */
