#include "ReverseToForwardNNF.cuh"

#include "../Utilities/Utilities.cuh"
#include "NNF.cuh"

#include <cuda_runtime.h>
#include <limits>
#include <stdio.h>

namespace StyLitCUDA {
namespace ReverseToForwardNNF {

int transfer(Image<NNFEntry> &reverse, Image<NNFEntry> &forward) {
  printf("StyLitCUDA: Transferring from reverse NNF with dimensions [%d %d] to forward NNF with "
         "dimensions [%d %d].\n",
         reverse.rows, reverse.cols, forward.rows, forward.cols);

  // Transfers the NNFs to host memory.
  const int reverseHostPitch = reverse.cols * sizeof(NNFEntry);
  const int reverseSizeInBytes = reverse.rows * reverseHostPitch;
  NNFEntry *reverseHost;
  check(cudaMallocHost(&reverseHost, reverseSizeInBytes));
  check(cudaMemcpy2D((void *)reverseHost, reverseHostPitch, reverse.deviceData, reverse.pitch,
                     reverseHostPitch, reverse.rows, cudaMemcpyDeviceToHost));

  const int forwardHostPitch = forward.cols * sizeof(NNFEntry);
  const int forwardSizeInBytes = forward.rows * forwardHostPitch;
  NNFEntry *forwardHost;
  check(cudaMallocHost(&forwardHost, forwardSizeInBytes));
  check(cudaMemcpy2D((void *)forwardHost, forwardHostPitch, forward.deviceData, forward.pitch,
                     forwardHostPitch, forward.rows, cudaMemcpyDeviceToHost));

  // TODO: Find a real knee point :)
  // For now, the knee point is based on the average of the valid mappings.
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
  check(cudaMemcpy2D((void *)reverse.deviceData, reverse.pitch, reverseHost, reverseHostPitch,
                     reverseHostPitch, reverse.rows, cudaMemcpyDeviceToHost));
  check(cudaFreeHost(reverseHost));

  check(cudaMemcpy2D((void *)forward.deviceData, forward.pitch, forwardHost, forwardHostPitch,
                     forwardHostPitch, forward.rows, cudaMemcpyDeviceToHost));
  check(cudaFreeHost(forwardHost));
  return numTransfers;
}

} /* namespace ReverseToForwardNNF */
} /* namespace StyLitCUDA */
