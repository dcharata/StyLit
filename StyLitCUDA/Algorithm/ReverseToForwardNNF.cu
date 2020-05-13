#include "ReverseToForwardNNF.cuh"

#include "../Utilities/Utilities.cuh"
#include "NNF.cuh"

#include <array>
#include <cuda_runtime.h>
#include <limits>
#include <stdio.h>

namespace StyLitCUDA {
namespace ReverseToForwardNNF {

/**
 * @brief memoryToHost Allocates space for the NNF's contents at hostNNF and transfers the NNF's
 * contents to hostNNF. Remember to call memoryToDevice after using memoryToHost in order to
 * transfer the modified memory back and free it!
 * @param nnf the NNF to transfer to host
 * @param hostNNF a pointer which will point to the on-host NNF data
 */
void memoryToHost(Image<NNFEntry> &nnf, NNFEntry *&hostNNF) {
  // Transfers the NNF to host memory.
  const int hostPitch = nnf.cols * sizeof(NNFEntry);
  const int sizeInBytes = nnf.rows * hostPitch;
  check(cudaMallocHost(&hostNNF, sizeInBytes));
  check(cudaMemcpy2D((void *)hostNNF, hostPitch, nnf.deviceData, nnf.pitch, hostPitch, nnf.rows,
                     cudaMemcpyDeviceToHost));
}

/**
 * @brief memoryToHost Copies the data from hostNNF to the on-device memory for the NNF, then frees
 * hostNNF.
 * @param nnf the NNF to transfer to
 * @param hostNNF a pointer which points to on-device NNF data allocated via memoryToHost
 * (cudaMallocHost)
 */
void memoryToDevice(Image<NNFEntry> &nnf, NNFEntry *&hostNNF) {
  const int hostPitch = nnf.cols * sizeof(NNFEntry);
  check(cudaMemcpy2D((void *)nnf.deviceData, nnf.pitch, hostNNF, hostPitch, hostPitch, nnf.rows,
                     cudaMemcpyDeviceToHost));
  check(cudaFreeHost(hostNNF));
  hostNNF = nullptr;
}

int transfer(Image<NNFEntry> &reverse, Image<NNFEntry> &forward) {
  printf("StyLitCUDA: Transferring from reverse NNF with dimensions [%d %d] to forward NNF with "
         "dimensions [%d %d].\n",
         reverse.rows, reverse.cols, forward.rows, forward.cols);

  // Transfers the NNFs to host memory.
  NNFEntry *reverseHost;
  NNFEntry *forwardHost;
  memoryToHost(reverse, reverseHost);
  memoryToHost(forward, forwardHost);

  // TODO: Find a real knee point :)
  // Super simple fake knee point based on max:
  /*float max = 0;
  for (int row = 0; row < reverse.rows; row++) {
    for (int col = 0; col < reverse.cols; col++) {
      const NNFEntry *entry = &reverseHost[row * reverse.cols + col];
      if (entry->error < 0.1f * std::numeric_limits<float>::max() && entry->error > max) {
        max = entry->error;
      }
    }
  }
  const float kneePoint = max * 0.01f;*/

  // Fake knee point based on mean:
  double sum = 0.;
  for (int row = 0; row < reverse.rows; row++) {
    for (int col = 0; col < reverse.cols; col++) {
      const NNFEntry *entry = &reverseHost[row * reverse.cols + col];
      if (entry->error < std::numeric_limits<float>::max()) {
        sum += (double)entry->error;
      }
    }
  }
  const float kneePoint = 3.f * (float)(sum / (reverse.rows * reverse.cols));

  // Transfers all mappings whose error is below the knee point.
  int numTransfers = 0;
  for (int row = 0; row < reverse.rows; row++) {
    for (int col = 0; col < reverse.cols; col++) {
      NNFEntry *reverseEntry = &reverseHost[row * reverse.cols + col];
      NNFEntry *forwardEntry = &forwardHost[reverseEntry->row * forward.cols + reverseEntry->col];

      // Does not assign the mapping if its error is above the knee point or the spot it maps to
      // already has an assignment.
      if (reverseEntry->error > kneePoint || forwardEntry->row != NNF::INVALID ||
          forwardEntry->col != NNF::INVALID) {
        continue;
      }

      // Updates the forward mapping.
      forwardEntry->error = reverseEntry->error;
      forwardEntry->row = row;
      forwardEntry->col = col;

      // Invalidates the reverse mapping so it will be changed during the next iteration of
      // PatchMatch.
      reverseEntry->error = std::numeric_limits<float>::max();
      numTransfers++;
    }
  }
  printf("StyLitCUDA: Transferred %d patches (%f percent of forward NNF).\n", numTransfers,
         (float)numTransfers / (forward.rows * forward.cols) * 100.f);

  // Copies the NNFs back.
  memoryToDevice(reverse, reverseHost);
  memoryToDevice(forward, forwardHost);
  return numTransfers;
}

void fill(Image<NNFEntry> &from, Image<NNFEntry> &to) {
  printf("StyLitCUDA: Filling in missing entries in NNF with dimensions [%d %d].\n", from.rows,
         from.cols);

  // Transfers the NNFs to host memory.
  NNFEntry *fromHost;
  NNFEntry *toHost;
  memoryToHost(from, fromHost);
  memoryToHost(to, toHost);
  int numFilled = 0;
  for (int row = 0; row < from.rows; row++) {
    for (int col = 0; col < from.cols; col++) {
      const int index = row * from.cols + col;
      const NNFEntry *fromEntry = &fromHost[index];
      NNFEntry *toEntry = &toHost[index];
      if (toEntry->row == NNF::INVALID || toEntry->col == NNF::INVALID) {
        numFilled++;
        toEntry->row = fromEntry->row;
        toEntry->col = fromEntry->col;
        toEntry->error = fromEntry->error;
      }
    }
  }
  const float percentFilled = (100.f * numFilled) / (from.rows * from.cols);
  printf("StyLitCUDA: Filled in %d entries (%f percent).\n", numFilled, percentFilled);

  // Copies the NNFs back.
  memoryToDevice(from, fromHost);
  memoryToDevice(to, toHost);
}

} /* namespace ReverseToForwardNNF */
} /* namespace StyLitCUDA */
