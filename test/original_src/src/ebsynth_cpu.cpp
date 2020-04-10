// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#include "ebsynth.h"
#include "jzq.h"

#include <cfloat>
#include <cmath>
#include <cstring>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#else
#include <omp.h>
#endif

// This is used to iterate over NNFs.
#define FOR(A, X, Y)                                                           \
  for (int Y = 0; Y < A.height(); Y++)                                         \
    for (int X = 0; X < A.width(); X++)

// This is completely unused.
A2V2i nnfInit(const V2i &size_of_target_B, const V2i &size_of_source_A, const int patchWidth) {
  A2V2i NNF(size_of_target_B);

  for (int xy = 0; xy < NNF.numel(); xy++) {
    NNF[xy] = V2i(patchWidth + rand() % (size_of_source_A(0) - 2 * patchWidth),
                  patchWidth + rand() % (size_of_source_A(1) - 2 * patchWidth));
  }

  return NNF;
}

// Runs PatchSSD_Split's operator() for each pixel to calculate error.
// PatchSSD_Split consists of A, A', B and B', so this is calculating E for each NNF mapping.
template <typename FUNC>
A2f nnfError(const A2V2i &NNF, const int patchWidth, FUNC patchError) {
  A2f E(size(NNF));

#pragma omp parallel for schedule(static)
  for (int y = 0; y < NNF.height(); y++)
    for (int x = 0; x < NNF.width(); x++) {
      // This is PatchSSD_Split's operator(), which calculates the error.
      // FLT_MAX is the highest non-infinity float value.
      E(x, y) = patchError(patchWidth, V2i(x, y), NNF(x, y), FLT_MAX);
    }

  return E;
}

// Initializes the NNF to point to random places in the source.
// Remember that the NNF is a target-sized 2D array of indices in the source.
static A2V2i nnfInitRandom(const V2i &targetSize, const V2i &sourceSize,
                           const int patchSize) {
  A2V2i NNF(targetSize);
  const int r = patchSize / 2;

  for (int i = 0; i < NNF.numel(); i++) {
    NNF[i] = V2i(r + (rand() % (sourceSize[0] - 2 * r)),
                 r + (rand() % (sourceSize[1] - 2 * r)));
  }

  return NNF;
}

// Upscales an NNF.
// This does what you would expect, except NNF values are clamped to be at least patchSize away from a border.
static A2V2i nnfUpscale(const A2V2i &NNF, const int patchSize,
                        const V2i &targetSize, const V2i &sourceSize) {
  A2V2i NNF2x(targetSize);
  // Iterates over the new NNF and populates it with temporary values.
  FOR(NNF2x, x, y) {
    // Finds the corresponding value in the half-resolution NNF, multiplies it by 2, and adds one to odd dimensions.
    NNF2x(x, y) = NNF(clamp(x / 2, 0, NNF.width() - 1),
                      clamp(y / 2, 0, NNF.height() - 1)) *
                      2 +
                  V2i(x % 2, y % 2);
  }

  // Converts the intermediate NNF value to a final NNF value.
  // This just clamps inwards by patch size.
  FOR(NNF2x, x, y) {
    const V2i nn = NNF2x(x, y);

    NNF2x(x, y) = V2i(clamp(nn(0), patchSize, sourceSize(0) - patchSize - 1),
                      clamp(nn(1), patchSize, sourceSize(1) - patchSize - 1));
  }

  return NNF2x;
}

// Populates the target B' using the source A' and the NNF.
// Each source sample is taken in a patch-sized square area.
// The weight is equal across the n * n sample area (no Gaussian kernel).
// Each t
// target - B'
// source - A'
// NNF - maps B/B' to A/A' (it's a 2D array of indices in A that has the same
// size a B) patchSize - the size of the sample in source which is
template <int N, typename T>
void krnlVotePlain(Array2<Vec<N, T>> &target, const Array2<Vec<N, T>> &source,
                   const Array2<Vec<2, int>> &NNF, const int patchSize) {
  // Iterates over each pixel in the target.
  for (int y = 0; y < target.height(); y++)
    for (int x = 0; x < target.width(); x++) {
      // For each pixel, keep track of a sum (sumColor) and a normalizing factor
      // (sumWeight).
      Vec<N, float> sumColor = zero<Vec<N, float>>::value();
      float sumWeight = 0;

      // Iterates over a patch-sized region.
      const int r = patchSize / 2;
      for (int py = -r; py <= +r; py++) {
        for (int px = -r; px <= +r; px++) {
          // Skips the pixel if it's outside of the NNF.
          // Remember: The NNF is target-sized, meaning that it maps target
          // pixels to source pixels.
          if (x + px >= 0 && x + px < NNF.width() && y + py >= 0 &&
              y + py < NNF.height()) {
            // n is the corresponding pixel in the source.
            // Note that it's offset by the patch position.
            const V2i n = NNF(x + px, y + py) - V2i(px, py);

            // Skips the pixel if the corresponding source pixel is outside the
            // source image.
            if (n[0] >= 0 && n[0] < source.width() && n[1] >= 0 &&
                n[1] < source.height()) {
              // Adds to the weighted average.
              const float weight = 1.0f;
              sumColor += weight * Vec<N, float>(source(n(0), n(1)));
              sumWeight += weight;
            }
          }
        }
      }

      // Sets the weighted average.
      const Vec<N, T> v = Vec<N, T>(sumColor / sumWeight);
      target(x, y) = v;
    }
}

#if 0
template<int N, typename T, int M>
__global__ void krnlVoteWeighted(      TexArray2<N,T,M>   target,
                                 const TexArray2<N,T,M>   source,
                                 const TexArray2<2,int>   NNF,
                                 const TexArray2<1,float> E,
                                 const int patchSize)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<target.width && y<target.height)
  {
    const int r = patchSize / 2;

    Vec<N,float> sumColor = zero<Vec<N,float>>::value();
    float sumWeight = 0;

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      /*
      if
      (
        x+px >= 0 && x+px < NNF.width () &&
        y+py >= 0 && y+py < NNF.height()
      )
      */
      {
        const V2i n = NNF(x+px,y+py)-V2i(px,py);

        /*if
        (
          n[0] >= 0 && n[0] < S.width () &&
          n[1] >= 0 && n[1] < S.height()
        )*/
        {
          const float error = E(x+px,y+py)(0)/(patchSize*patchSize*N);
          const float weight = 1.0f/(1.0f+error);
          sumColor += weight*Vec<N,float>(source(n(0),n(1)));
          sumWeight += weight;
        }
      }
    }

    const Vec<N,T> v = Vec<N,T>(sumColor/sumWeight);
    target.write(x,y,v);
  }
}
#endif

template <int N, typename T>
Vec<N, T> sampleBilinear(const Array2<Vec<N, T>> &I, float x, float y) {
  const int ix = x;
  const int iy = y;

  const float s = x - ix;
  const float t = y - iy;

  return Vec<N, T>(
      (1.0f - s) * (1.0f - t) *
          Vec<N, float>(
              I(clamp(ix, 0, I.width() - 1), clamp(iy, 0, I.height() - 1))) +
      (s) * (1.0f - t) *
          Vec<N, float>(I(clamp(ix + 1, 0, I.width() - 1),
                          clamp(iy, 0, I.height() - 1))) +
      (1.0f - s) * (t)*Vec<N, float>(I(clamp(ix, 0, I.width() - 1),
                                       clamp(iy + 1, 0, I.height() - 1))) +
      (s) * (t)*Vec<N, float>(I(clamp(ix + 1, 0, I.width() - 1),
                                clamp(iy + 1, 0, I.height() - 1))));
};

template <int N, typename T>
void resampleCPU(Array2<Vec<N, T>> &O, const Array2<Vec<N, T>> &I) {
  const float s = float(I.width()) / float(O.width());

  for (int y = 0; y < O.height(); y++)
    for (int x = 0; x < O.width(); x++) {
      O(x, y) = sampleBilinear(I, s * float(x), s *float(y));
    }
}


template <int NS, int NG, typename T> struct PatchSSD_Split {
  const Array2<Vec<NS, T>> &targetStyle;
  const Array2<Vec<NS, T>> &sourceStyle;

  const Array2<Vec<NG, T>> &targetGuide;
  const Array2<Vec<NG, T>> &sourceGuide;

  const Vec<NS, float> &styleWeights;
  const Vec<NG, float> &guideWeights;

  // It's confusingly formatted, but this is a constructor.
  PatchSSD_Split(const Array2<Vec<NS, T>> &targetStyle,
                 const Array2<Vec<NS, T>> &sourceStyle,

                 const Array2<Vec<NG, T>> &targetGuide,
                 const Array2<Vec<NG, T>> &sourceGuide,

                 const Vec<NS, float> &styleWeights,
                 const Vec<NG, float> &guideWeights)

      : targetStyle(targetStyle), sourceStyle(sourceStyle),
        targetGuide(targetGuide), sourceGuide(sourceGuide),
        styleWeights(styleWeights), guideWeights(guideWeights) {}

  // This is the operator () of PatchSSD_Split.
  // It calculates error for the PatchSSD_Split by adding up the error over a patch-sized region centered at txy.
  // Error is difference between A' and B' or A and B squared and multiplied by a weight.
  // Remember that the NNF is a target-sized (B/B'-sized) array of indices in the source A/A'.
  float operator()(const int patchSize, const V2i txy, const V2i sxy,
                   const float ebest) {
    // These are the coordinates for which the NNF error is calculated.
    const int tx = txy(0);
    const int ty = txy(1);

    // This is the NNF's value at the above coordinates.
    const int sx = sxy(0);
    const int sy = sxy(1);

    const int r = patchSize / 2;
    float error = 0;

    if (tx - r >= 0 && tx + r < targetStyle.width() && ty - r >= 0 &&
        ty + r < targetStyle.height()) {
      // This branch executes if a patch centered at the coordinates is within the target/NNF.
      // These are pointers for pixels in A, A', B, B'.
      // Note that they're const T *, but not const T *const, so the pointers themselves can change, but the underlying values can't.
      const T *pixel_in_B_prime = (T *)&targetStyle(tx - r, ty - r); // B'
      const T *pixel_in_A_prime = (T *)&sourceStyle(sx - r, sy - r); // A
      const T *pixel_in_B = (T *)&targetGuide(tx - r, ty - r);
      const T *pixel_in_A = (T *)&sourceGuide(sx - r, sy - r);

      // NS is the number of style channels, and NG is the number of guidance channels.
      // These offsets are used to skip an entire row in A, A', B and B'.
      const int bytes_per_row_in_B_prime = (targetStyle.width() - patchSize) * NS;
      const int bytes_per_row_in_A_prime = (sourceStyle.width() - patchSize) * NS;
      const int bytes_per_row_in_B = (targetGuide.width() - patchSize) * NG;
      const int bytes_per_row_in_A = (sourceGuide.width() - patchSize) * NG;

      // Iterates over a patch.
      for (int j = 0; j < patchSize; j++) {
        for (int i = 0; i < patchSize; i++) {
          // Each error is the difference squared of a layer's value times the layer's weight.
          // Note: Each channel can have between 1 and 4 layers (see ebsynth.cpp).
          for (int k = 0; k < NS; k++) {
            const float diff = *pixel_in_B_prime - *pixel_in_A_prime;
            error += styleWeights[k] * diff * diff;
            pixel_in_B_prime++;
            pixel_in_A_prime++;
          }
          for (int k = 0; k < NG; k++) {
            const float diff = *pixel_in_B - *pixel_in_A;
            error += guideWeights[k] * diff * diff;
            pixel_in_B++;
            pixel_in_A++;
          }
        }

        // Uses direct pointer math to go to the next row in the patch.
        pixel_in_B_prime += bytes_per_row_in_B_prime;
        pixel_in_A_prime += bytes_per_row_in_A_prime;
        pixel_in_B += bytes_per_row_in_B;
        pixel_in_A += bytes_per_row_in_A;

        // ebest is the highest possible float value.
        // In other words, this halts the calculation once the error reaches infinity.
        if (error > ebest) {
          break;
        }
      }
    } else {
      // This handles the case where a patch centered at txy would go beyond the edge of the NNF.
      // This is essentially the same as the above case, except the values of txy are clamped to be within the NNF.
      for (int py = -r; py <= +r; py++)
        for (int px = -r; px <= +r; px++) {
          {
            const Vec<NS, T> pixTs =
                targetStyle(clamp(tx + px, 0, targetStyle.width() - 1),
                            clamp(ty + py, 0, targetStyle.height() - 1));
            const Vec<NS, T> pixSs =
                sourceStyle(clamp(sx + px, 0, sourceStyle.width() - 1),
                            clamp(sy + py, 0, sourceStyle.height() - 1));
            for (int i = 0; i < NS; i++) {
              const float diff = float(pixTs[i]) - float(pixSs[i]);
              error += styleWeights[i] * diff * diff;
            }
          }

          {
            const Vec<NG, T> pixTg =
                targetGuide(clamp(tx + px, 0, targetGuide.width() - 1),
                            clamp(ty + py, 0, targetGuide.height() - 1));
            const Vec<NG, T> pixSg =
                sourceGuide(clamp(sx + px, 0, sourceGuide.width() - 1),
                            clamp(sy + py, 0, sourceGuide.height() - 1));
            for (int i = 0; i < NG; i++) {
              const float diff = float(pixTg[i]) - float(pixSg[i]);
              error += guideWeights[i] * diff * diff;
            }
          }
        }
    }

    return error;
  }
};

static V2i pyramidLevelSize(const V2i &sizeBase, const int numLevels,
                            const int level) {
  return V2i(V2f(sizeBase) * std::pow(2.0f, -float(numLevels - 1 - level)));
}

template <typename T> void copy(Array2<T> *out_dst, void *src) {
  Array2<T> &dst = *out_dst;
  memcpy(dst.data(), src, numel(dst) * sizeof(T));
}

template <typename T> void copy(void **out_dst, const Array2<T> &src) {
  void *&dst = *out_dst;
  memcpy(dst, src.data(), numel(src) * sizeof(T));
}

// This is called from PatchMatch. It increments the omega values in a patch-sized region around nnf_value by incdec.
// The size of omega is the same as the size of the source.
// It had an unused variable called axy.
// Most of this is just array iteration doing direct pointer math.
void updateOmega(A2i &Omega, const V2i &size_of_target_B, const int patchWidth,
                 const V2i &, const V2i &nnf_value, const int incdec) {
  const int r = patchWidth / 2;
  int *ptr = (int *)&Omega(nnf_value(0) - r, nnf_value(1) - r);
  const int ofs = (Omega.width() - patchWidth);
  for (int j = 0; j < patchWidth; j++) {
    for (int i = 0; i < patchWidth; i++) {
      *ptr += incdec;
      ptr++;
    }
    ptr += ofs;
  }
}

static int patchOmega(const int patchWidth, const V2i &bxy, const A2i &Omega) {
  const int r = patchWidth / 2;

  int sum = 0;

  const int *ptr = (int *)&Omega(bxy(0) - r, bxy(1) - r);
  const int ofs = (Omega.width() - patchWidth);

  for (int j = 0; j < patchWidth; j++) {
    for (int i = 0; i < patchWidth; i++) {
      sum += (*ptr);
      ptr++;
    }
    ptr += ofs;
  }

  return sum;
}

template <typename FUNC>
bool tryPatch(FUNC patchError, const V2i &size_of_target_B, int patchWidth, const V2i &axy,
              const V2i &bxy, A2V2i &N, A2f &E, A2i &Omega, float omegaBest,
              float lambda) {
  const float curOcc = (float(patchOmega(patchWidth, N(axy), Omega)) /
                        float(patchWidth * patchWidth)) /
                       omegaBest;
  const float newOcc = (float(patchOmega(patchWidth, bxy, Omega)) /
                        float(patchWidth * patchWidth)) /
                       omegaBest;

  const float curErr = E(axy);
  const float newErr =
      patchError(patchWidth, axy, bxy, curErr + lambda * curOcc);

  if ((newErr + lambda * newOcc) < (curErr + lambda * curOcc)) {
    updateOmega(Omega, size_of_target_B, patchWidth, axy, bxy, +1);
    updateOmega(Omega, size_of_target_B, patchWidth, axy, N(axy), -1);
    N(axy) = bxy;
    E(axy) = newErr;
  }

  return true;
}

// This is the patch match algorithm that is run some number of times for each pixel at each pyramid level.
// size_of_target_B - I renamed this from sizeA because it is in fact not the size of the source A/A', but rather the size of the target B/B' and the NNF.
// size_of_source_A - I also renamed this to make sense.
// FUNC is PatchSSD_Split, which is a struct whose operator() calculates error for a particular NNF location.
template <typename FUNC>
void patchmatch(const V2i &size_of_target_B, const V2i &size_of_source_A, const int patchWidth,
                FUNC patchError, const float lambda, const int numIters,
                const int numThreads, A2V2i &N, A2f &E, A2i &Omega) {
  // There's no reason why this should exist because both patchWidth and w are const.
  const int w = patchWidth;

  // Calculates error for all NNF pixels.
  E = nnfError(N, patchWidth, patchError);

  // Pushes the larger one of source's dimensions to irad.
  // I do not understand how the while loop will ever break without overflowing.
  // irad[0] will be a value > 1, and irad.size() will be >= 1, so int(std::pow(0.5, number > 1)) will be zero...
  // Anyway, nir is some number of iterations based on the source's size.
  std::vector<int> irad;
  irad.push_back((size_of_source_A(0) > size_of_source_A(1) ? size_of_source_A(0) : size_of_source_A(1)));
  const float sra = 0.5f;
  while (irad.back() != 1)
    irad.push_back(int(std::pow(sra, int(irad.size())) * irad[0]));
  const int nir = int(irad.size());

#ifdef __APPLE__
  dispatch_queue_t gcdq =
      dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
  const int numThreads_ = 8;
#else
  const int numThreads_ = numThreads < 1 ? omp_get_max_threads() : numThreads;
#endif

  // Splits the target into a number of tiles. tileHeight is the number of tiles.
  const int minTileHeight = 8;
  const int numTiles =
      int(ceil(float(size_of_target_B(1)) / float(numThreads_))) > minTileHeight
          ? numThreads_
          : std::max(int(ceil(float(size_of_target_B(1)) / float(minTileHeight))), 1);
  const int tileHeight = size_of_target_B(1) / numTiles;

  // This is (target area / source area) * patch area.
  const float omegaBest =
      (float(size_of_target_B(0) * size_of_target_B(1)) / float(size_of_source_A(0) * size_of_source_A(1))) *
      float(patchWidth * patchWidth);

  // Calculates omega for each pixel in the target B.
  // Outside of the edge cases, this updates all omega values to be a patch's area.
  // Because of how NNF is constructed, this might have somewhat weird values near the edges, not sure.
  fill(&Omega, (int)0);
  for (int y = 0; y < size_of_target_B(1); y++) {
    for (int x = 0; x < size_of_target_B(0); x++) {
      updateOmega(Omega, size_of_target_B, w, V2i(x, y), N(x, y), +1);
    }
  }

  // Runs a number of iterations that can vary by level.
  for (int iter = 0; iter < numIters; iter++) {
    const int iter_seed = rand();

// Starts a number of threads that's equal to numTiles.
#ifdef __APPLE__
    dispatch_apply(
        numTiles, gcdq,
        ^(size_t blockIdx)
#else
#pragma omp parallel num_threads(numTiles)
#endif
        {
          const bool odd = (iter % 2 == 0);

#ifdef __APPLE__
          const int threadId = blockIdx;
#else
      const int threadId = omp_get_thread_num();
#endif

          const int _y0 = threadId * tileHeight;
          const int _y1 = threadId == numTiles - 1
                              ? size_of_target_B(1)
                              : std::min(_y0 + tileHeight, size_of_target_B(1));

          const int q = odd ? 1 : -1;
          const int x0 = odd ? 0 : size_of_target_B(0) - 1;
          const int y0 = odd ? _y0 : _y1 - 1;
          const int x1 = odd ? size_of_target_B(0) : -1;
          const int y1 = odd ? _y1 : _y0 - 1;

          for (int y = y0; y != y1; y += q)
            for (int x = x0; x != x1; x += q) {
              if (odd ? (x > 0) : (x < size_of_target_B(0) - 1)) {
                V2i n = N(x - q, y);
                n[0] += q;

                if (odd ? (n[0] < size_of_source_A(0) - w / 2) : (n[0] >= w / 2)) {
                  tryPatch(patchError, size_of_target_B, w, V2i(x, y), n, N, E, Omega,
                           omegaBest, lambda);
                }
              }

              if (odd ? (y > 0) : (y < size_of_target_B(1) - 1)) {
                V2i n = N(x, y - q);
                n[1] += q;

                if (odd ? (n[1] < size_of_source_A(1) - w / 2) : (n[1] >= w / 2)) {
                  tryPatch(patchError, size_of_target_B, w, V2i(x, y), n, N, E, Omega,
                           omegaBest, lambda);
                }
              }

#define RANDI(u) (18000 * ((u)&65535) + ((u) >> 16))

              unsigned int seed = (x | (y << 11)) ^ iter_seed;
              seed = RANDI(seed);

              const V2i pix0 = N(x, y);
              // for (int i = 0; i < nir; i++)
              for (int i = nir - 1; i >= 0; i--) {
                V2i tl = pix0 - V2i(irad[i], irad[i]);
                V2i br = pix0 + V2i(irad[i], irad[i]);

                tl = std::max(tl, V2i(w / 2, w / 2));
                br = std::min(br, size_of_source_A - V2i(w / 2, w / 2));

                const int _rndX = RANDI(seed);
                const int _rndY = RANDI(_rndX);
                seed = _rndY;

                const V2i n = V2i(tl[0] + (_rndX % (br[0] - tl[0])),
                                  tl[1] + (_rndY % (br[1] - tl[1])));

                tryPatch(patchError, size_of_target_B, w, V2i(x, y), n, N, E, Omega,
                         omegaBest, lambda);
              }

#undef RANDI
            }
        }
#ifdef __APPLE__
    );
#endif
  }
}

template <int NS, int NG>
void ebsynthCpu(int numStyleChannels, int numGuideChannels, int sourceWidth,
                int sourceHeight, void *sourceStyleData, void *sourceGuideData,
                int targetWidth, int targetHeight, void *targetGuideData,
                void *targetModulationData, float *styleWeights,
                float *guideWeights, float uniformityWeight, int patchSize,
                int voteMode, int numPyramidLevels,
                int *numSearchVoteItersPerLevel,
                int *numPatchMatchItersPerLevel, int *stopThresholdPerLevel,
                int extraPass3x3, void *outputNnfData, void *outputImageData) {
  // There's no reason why this is a separate variable (ctrl + f to verify).
  const int levelCount = numPyramidLevels;

  struct PyramidLevel {
    PyramidLevel() {}

    int sourceWidth;
    int sourceHeight;
    int targetWidth;
    int targetHeight;

    Array2<Vec<NS, unsigned char>> sourceStyle;  // A'
    Array2<Vec<NG, unsigned char>> sourceGuide;  // B
    Array2<Vec<NS, unsigned char>> targetStyle;  // B'
    Array2<Vec<NS, unsigned char>> targetStyle2; // also B'
    Array2<Vec<NG, unsigned char>> targetGuide;  // A
    Array2<Vec<NG, unsigned char>> targetModulation;
    Array2<Vec<2, int>> NNF;

    // This stuff probably PatchMatch
    Array2<float> E;
    Array2<int> Omega;
  };

  // Calculates the size for each pyramid level.
  std::vector<PyramidLevel> pyramid(levelCount);
  for (int level = 0; level < levelCount; level++) {
    const V2i levelSourceSize =
        pyramidLevelSize(V2i(sourceWidth, sourceHeight), levelCount, level);
    const V2i levelTargetSize =
        pyramidLevelSize(V2i(targetWidth, targetHeight), levelCount, level);

    pyramid[level].sourceWidth = levelSourceSize(0);
    pyramid[level].sourceHeight = levelSourceSize(1);
    pyramid[level].targetWidth = levelTargetSize(0);
    pyramid[level].targetHeight = levelTargetSize(1);
  }

  // Allocates and copies over the finest pyramid level's A, A' and B.
  pyramid[levelCount - 1].sourceStyle =
      Array2<Vec<NS, unsigned char>>(V2i(pyramid[levelCount - 1].sourceWidth,
                                         pyramid[levelCount - 1].sourceHeight));
  pyramid[levelCount - 1].sourceGuide =
      Array2<Vec<NG, unsigned char>>(V2i(pyramid[levelCount - 1].sourceWidth,
                                         pyramid[levelCount - 1].sourceHeight));
  pyramid[levelCount - 1].targetGuide =
      Array2<Vec<NG, unsigned char>>(V2i(pyramid[levelCount - 1].targetWidth,
                                         pyramid[levelCount - 1].targetHeight));
  copy(&pyramid[levelCount - 1].sourceStyle, sourceStyleData);
  copy(&pyramid[levelCount - 1].sourceGuide, sourceGuideData);
  copy(&pyramid[levelCount - 1].targetGuide, targetGuideData);

  // ???
  if (targetModulationData) {
    pyramid[levelCount - 1].targetModulation = Array2<Vec<NG, unsigned char>>(
        V2i(pyramid[levelCount - 1].targetWidth,
            pyramid[levelCount - 1].targetHeight));
    copy(&pyramid[levelCount - 1].targetModulation, targetModulationData);
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Iterates through the pyramid levels.
  // The number of iterations could be pyramid.size() if there's an extra pass.
  // If there's an extra pass, the loop variable is messed with to get an extra
  // for loop iteration.
  bool inExtraPass = false;
  for (int level = 0; level < pyramid.size(); level++) {
    if (!inExtraPass) {
      // Allocates the arrays needed for the level's calculations.
      const V2i levelSourceSize =
          V2i(pyramid[level].sourceWidth, pyramid[level].sourceHeight);
      const V2i levelTargetSize =
          V2i(pyramid[level].targetWidth, pyramid[level].targetHeight);
      pyramid[level].targetStyle =
          Array2<Vec<NS, unsigned char>>(levelTargetSize);
      pyramid[level].targetStyle2 =
          Array2<Vec<NS, unsigned char>>(levelTargetSize);
      pyramid[level].NNF = Array2<Vec<2, int>>(levelTargetSize);
      pyramid[level].Omega = Array2<int>(levelSourceSize);
      pyramid[level].E = Array2<float>(levelTargetSize);

      if (level < levelCount - 1) {
        // The source style, source guide and target guide aren't initialized at
        // the last level.
        pyramid[level].sourceStyle =
            Array2<Vec<NS, unsigned char>>(levelSourceSize);
        pyramid[level].sourceGuide =
            Array2<Vec<NG, unsigned char>>(levelSourceSize);
        pyramid[level].targetGuide =
            Array2<Vec<NG, unsigned char>>(levelTargetSize);

        // The source style, source guide and target guide are upscaled from the
        // previous level.
        resampleCPU(pyramid[level].sourceStyle,
                    pyramid[levelCount - 1].sourceStyle);
        resampleCPU(pyramid[level].sourceGuide,
                    pyramid[levelCount - 1].sourceGuide);
        resampleCPU(pyramid[level].targetGuide,
                    pyramid[levelCount - 1].targetGuide);

        // ???
        if (targetModulationData) {
          resampleCPU(pyramid[level].targetModulation,
                      pyramid[levelCount - 1].targetModulation);
          pyramid[level].targetModulation =
              Array2<Vec<NG, unsigned char>>(levelTargetSize);
        }
      }

      // Upscales the NNF if above the lowest level.
      A2V2i cpu_NNF;
      if (level > 0) {
        pyramid[level].NNF = nnfUpscale(
            pyramid[level - 1].NNF, patchSize,
            V2i(pyramid[level].targetWidth, pyramid[level].targetHeight),
            V2i(pyramid[level].sourceWidth, pyramid[level].sourceHeight));

        pyramid[level - 1].NNF = A2V2i();
      } else {
        pyramid[level].NNF = nnfInitRandom(
            V2i(pyramid[level].targetWidth, pyramid[level].targetHeight),
            V2i(pyramid[level].sourceWidth, pyramid[level].sourceHeight),
            patchSize);
      }
    }

    // Uses the source style A' and NNF to populate the target style B'.
    // Also places the new target style B' into targetStyle (not targetStyle2).
    {
      krnlVotePlain(pyramid[level].targetStyle2, pyramid[level].sourceStyle,
                    pyramid[level].NNF, patchSize);
      std::swap(pyramid[level].targetStyle2, pyramid[level].targetStyle);
    }

    // Runs PatchMatch for 6 iterations.
    for (int voteIter = 0; voteIter < numSearchVoteItersPerLevel[level];
         voteIter++) {
      // Puts the style and guide weights into vectors.
      // Note: The style and guide weights determine how much each layer of each
      // channel is weighted. The layers and their weights are calculated in
      // ebsynth.cpp, where the program detects if a channel has alpha and
      // whether it's grayscale.
      Vec<NS, float> styleWeightsVec;
      for (int i = 0; i < NS; i++) {
        styleWeightsVec[i] = styleWeights[i];
      }
      Vec<NG, float> guideWeightsVec;
      for (int i = 0; i < NG; i++) {
        guideWeightsVec[i] = guideWeights[i];
      }

      // I'm not sure why function calls are wrapped in anonymous namespaces.
      // I removed the anonymous namespaces to improve readability.
      patchmatch(V2i(pyramid[level].targetWidth, pyramid[level].targetHeight),
                 V2i(pyramid[level].sourceWidth, pyramid[level].sourceHeight),
                 patchSize,

                 // A, A', B, B' and the weights are put into a PatchSSD_Split struct.
                 PatchSSD_Split<NS, NG, unsigned char>(
                     pyramid[level].targetStyle, pyramid[level].sourceStyle,
                     pyramid[level].targetGuide, pyramid[level].sourceGuide,
                     styleWeightsVec, guideWeightsVec),
                 uniformityWeight, numPatchMatchItersPerLevel[level], -1,
                 pyramid[level].NNF, pyramid[level].E, pyramid[level].Omega);

      // Uses the source style A' and NNF to populate the target style B'.
      // Also places the new target style B' into targetStyle (not
      // targetStyle2).
      krnlVotePlain(pyramid[level].targetStyle2, pyramid[level].sourceStyle,
                    pyramid[level].NNF, patchSize);
      std::swap(pyramid[level].targetStyle2, pyramid[level].targetStyle);
    }

    // Read the comment and if statement surrounding the line "inExtraPass =
    // true" to understand this if statement. This if branch is taken if the
    // current level is actually the last one (including the extra pass).
    if (level == levelCount - 1 &&
        (extraPass3x3 == 0 || (extraPass3x3 != 0 && inExtraPass))) {
      // It looks like this allows the NNF data to be exported for debugging.
      if (outputNnfData != NULL) {
        copy(&outputNnfData, pyramid[level].NNF);
      }

      // Copies the final target image to outputImageData.
      copy(&outputImageData, pyramid[level].targetStyle);
    }

    // This is NOT the same if condition as the previous if statement.
    // This resets the current pyramid level's arrays for all levels except the
    // last one.
    if ((level < levelCount - 1) || (extraPass3x3 == 0) ||
        (extraPass3x3 != 0 && inExtraPass)) {
      pyramid[level].sourceStyle = Array2<Vec<NS, unsigned char>>();
      pyramid[level].sourceGuide = Array2<Vec<NG, unsigned char>>();
      pyramid[level].targetGuide = Array2<Vec<NG, unsigned char>>();
      pyramid[level].targetStyle = Array2<Vec<NS, unsigned char>>();
      pyramid[level].targetStyle2 = Array2<Vec<NS, unsigned char>>();
      pyramid[level].Omega = Array2<int>();
      pyramid[level].E = Array2<float>();

      // ???
      if (targetModulationData) {
        pyramid[level].targetModulation = Array2<Vec<NG, unsigned char>>();
      }
    }

    // If extraPass3x3 is specified and this is the last pass, does another pass
    // with a 3x3 patch size. Warning: This messes with the pyramid level for
    // loop iterator.
    if (level == levelCount - 1 && (extraPass3x3 != 0) && !inExtraPass) {
      inExtraPass = true;
      level--;
      patchSize = 3;
      uniformityWeight = 0;
    }
  }

  pyramid[levelCount - 1].NNF = Array2<Vec<2, int>>();
}

// Start here!
// Picks the right template version of the function to run.
void ebsynthRunCpu(int numStyleChannels, int numGuideChannels, int sourceWidth,
                   int sourceHeight, void *sourceStyleData,
                   void *sourceGuideData, int targetWidth, int targetHeight,
                   void *targetGuideData, void *targetModulationData,
                   float *styleWeights, float *guideWeights,
                   float uniformityWeight, int patchSize, int voteMode,
                   int numPyramidLevels, int *numSearchVoteItersPerLevel,
                   int *numPatchMatchItersPerLevel, int *stopThresholdPerLevel,
                   int extraPass3x3, void *outputNnfData,
                   void *outputImageData) {
  // There must be a better way to set this up...
  void (*const dispatchEbsynth[EBSYNTH_MAX_GUIDE_CHANNELS]
                              [EBSYNTH_MAX_STYLE_CHANNELS])(
      int, int, int, int, void *, void *, int, int, void *, void *, float *,
      float *, float, int, int, int, int *, int *, int *, int, void *,
      void *) = {
      {ebsynthCpu<1, 1>, ebsynthCpu<2, 1>, ebsynthCpu<3, 1>, ebsynthCpu<4, 1>,
       ebsynthCpu<5, 1>, ebsynthCpu<6, 1>, ebsynthCpu<7, 1>, ebsynthCpu<8, 1>},
      {ebsynthCpu<1, 2>, ebsynthCpu<2, 2>, ebsynthCpu<3, 2>, ebsynthCpu<4, 2>,
       ebsynthCpu<5, 2>, ebsynthCpu<6, 2>, ebsynthCpu<7, 2>, ebsynthCpu<8, 2>},
      {ebsynthCpu<1, 3>, ebsynthCpu<2, 3>, ebsynthCpu<3, 3>, ebsynthCpu<4, 3>,
       ebsynthCpu<5, 3>, ebsynthCpu<6, 3>, ebsynthCpu<7, 3>, ebsynthCpu<8, 3>},
      {ebsynthCpu<1, 4>, ebsynthCpu<2, 4>, ebsynthCpu<3, 4>, ebsynthCpu<4, 4>,
       ebsynthCpu<5, 4>, ebsynthCpu<6, 4>, ebsynthCpu<7, 4>, ebsynthCpu<8, 4>},
      {ebsynthCpu<1, 5>, ebsynthCpu<2, 5>, ebsynthCpu<3, 5>, ebsynthCpu<4, 5>,
       ebsynthCpu<5, 5>, ebsynthCpu<6, 5>, ebsynthCpu<7, 5>, ebsynthCpu<8, 5>},
      {ebsynthCpu<1, 6>, ebsynthCpu<2, 6>, ebsynthCpu<3, 6>, ebsynthCpu<4, 6>,
       ebsynthCpu<5, 6>, ebsynthCpu<6, 6>, ebsynthCpu<7, 6>, ebsynthCpu<8, 6>},
      {ebsynthCpu<1, 7>, ebsynthCpu<2, 7>, ebsynthCpu<3, 7>, ebsynthCpu<4, 7>,
       ebsynthCpu<5, 7>, ebsynthCpu<6, 7>, ebsynthCpu<7, 7>, ebsynthCpu<8, 7>},
      {ebsynthCpu<1, 8>, ebsynthCpu<2, 8>, ebsynthCpu<3, 8>, ebsynthCpu<4, 8>,
       ebsynthCpu<5, 8>, ebsynthCpu<6, 8>, ebsynthCpu<7, 8>, ebsynthCpu<8, 8>},
      {ebsynthCpu<1, 9>, ebsynthCpu<2, 9>, ebsynthCpu<3, 9>, ebsynthCpu<4, 9>,
       ebsynthCpu<5, 9>, ebsynthCpu<6, 9>, ebsynthCpu<7, 9>, ebsynthCpu<8, 9>},
      {ebsynthCpu<1, 10>, ebsynthCpu<2, 10>, ebsynthCpu<3, 10>,
       ebsynthCpu<4, 10>, ebsynthCpu<5, 10>, ebsynthCpu<6, 10>,
       ebsynthCpu<7, 10>, ebsynthCpu<8, 10>},
      {ebsynthCpu<1, 11>, ebsynthCpu<2, 11>, ebsynthCpu<3, 11>,
       ebsynthCpu<4, 11>, ebsynthCpu<5, 11>, ebsynthCpu<6, 11>,
       ebsynthCpu<7, 11>, ebsynthCpu<8, 11>},
      {ebsynthCpu<1, 12>, ebsynthCpu<2, 12>, ebsynthCpu<3, 12>,
       ebsynthCpu<4, 12>, ebsynthCpu<5, 12>, ebsynthCpu<6, 12>,
       ebsynthCpu<7, 12>, ebsynthCpu<8, 12>},
      {ebsynthCpu<1, 13>, ebsynthCpu<2, 13>, ebsynthCpu<3, 13>,
       ebsynthCpu<4, 13>, ebsynthCpu<5, 13>, ebsynthCpu<6, 13>,
       ebsynthCpu<7, 13>, ebsynthCpu<8, 13>},
      {ebsynthCpu<1, 14>, ebsynthCpu<2, 14>, ebsynthCpu<3, 14>,
       ebsynthCpu<4, 14>, ebsynthCpu<5, 14>, ebsynthCpu<6, 14>,
       ebsynthCpu<7, 14>, ebsynthCpu<8, 14>},
      {ebsynthCpu<1, 15>, ebsynthCpu<2, 15>, ebsynthCpu<3, 15>,
       ebsynthCpu<4, 15>, ebsynthCpu<5, 15>, ebsynthCpu<6, 15>,
       ebsynthCpu<7, 15>, ebsynthCpu<8, 15>},
      {ebsynthCpu<1, 16>, ebsynthCpu<2, 16>, ebsynthCpu<3, 16>,
       ebsynthCpu<4, 16>, ebsynthCpu<5, 16>, ebsynthCpu<6, 16>,
       ebsynthCpu<7, 16>, ebsynthCpu<8, 16>},
      {ebsynthCpu<1, 17>, ebsynthCpu<2, 17>, ebsynthCpu<3, 17>,
       ebsynthCpu<4, 17>, ebsynthCpu<5, 17>, ebsynthCpu<6, 17>,
       ebsynthCpu<7, 17>, ebsynthCpu<8, 17>},
      {ebsynthCpu<1, 18>, ebsynthCpu<2, 18>, ebsynthCpu<3, 18>,
       ebsynthCpu<4, 18>, ebsynthCpu<5, 18>, ebsynthCpu<6, 18>,
       ebsynthCpu<7, 18>, ebsynthCpu<8, 18>},
      {ebsynthCpu<1, 19>, ebsynthCpu<2, 19>, ebsynthCpu<3, 19>,
       ebsynthCpu<4, 19>, ebsynthCpu<5, 19>, ebsynthCpu<6, 19>,
       ebsynthCpu<7, 19>, ebsynthCpu<8, 19>},
      {ebsynthCpu<1, 20>, ebsynthCpu<2, 20>, ebsynthCpu<3, 20>,
       ebsynthCpu<4, 20>, ebsynthCpu<5, 20>, ebsynthCpu<6, 20>,
       ebsynthCpu<7, 20>, ebsynthCpu<8, 20>},
      {ebsynthCpu<1, 21>, ebsynthCpu<2, 21>, ebsynthCpu<3, 21>,
       ebsynthCpu<4, 21>, ebsynthCpu<5, 21>, ebsynthCpu<6, 21>,
       ebsynthCpu<7, 21>, ebsynthCpu<8, 21>},
      {ebsynthCpu<1, 22>, ebsynthCpu<2, 22>, ebsynthCpu<3, 22>,
       ebsynthCpu<4, 22>, ebsynthCpu<5, 22>, ebsynthCpu<6, 22>,
       ebsynthCpu<7, 22>, ebsynthCpu<8, 22>},
      {ebsynthCpu<1, 23>, ebsynthCpu<2, 23>, ebsynthCpu<3, 23>,
       ebsynthCpu<4, 23>, ebsynthCpu<5, 23>, ebsynthCpu<6, 23>,
       ebsynthCpu<7, 23>, ebsynthCpu<8, 23>},
      {ebsynthCpu<1, 24>, ebsynthCpu<2, 24>, ebsynthCpu<3, 24>,
       ebsynthCpu<4, 24>, ebsynthCpu<5, 24>, ebsynthCpu<6, 24>,
       ebsynthCpu<7, 24>, ebsynthCpu<8, 24>}};

  // Makes sure the number of channels is covered by one of the above templates
  // (lol).
  if (numStyleChannels >= 1 && numStyleChannels <= EBSYNTH_MAX_STYLE_CHANNELS &&
      numGuideChannels >= 1 && numGuideChannels <= EBSYNTH_MAX_GUIDE_CHANNELS) {
    dispatchEbsynth[numGuideChannels - 1][numStyleChannels - 1](
        numStyleChannels, numGuideChannels, sourceWidth, sourceHeight,
        sourceStyleData, sourceGuideData, targetWidth, targetHeight,
        targetGuideData, targetModulationData, styleWeights, guideWeights,
        uniformityWeight, patchSize, voteMode, numPyramidLevels,
        numSearchVoteItersPerLevel, numPatchMatchItersPerLevel,
        stopThresholdPerLevel, extraPass3x3, outputNnfData, outputImageData);
  }
}

int ebsynthBackendAvailableCpu() { return 1; }
