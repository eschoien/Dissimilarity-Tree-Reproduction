#pragma once

#include <utility>
#include <shapeDescriptor/libraryBuildSettings.h>
#include <algorithm>
#include <bitset>

#ifndef __CUDACC__
unsigned int __popc(unsigned int x);
#endif

namespace ProjectSymmetry {
    namespace utilities {
        struct HammingWeights {
            float missingSetBitPenalty = 0;
            float missingUnsetBitPenalty = 0;
        };

#ifdef __CUDACC__
        __host__
            __device__
#endif
        inline HammingWeights computeWeightedHammingWeights(
                unsigned int setBitCount, unsigned int totalBitsInBitString) {
            unsigned int queryImageUnsetBitCount = totalBitsInBitString - setBitCount;

            // If any count is 0, bump it up to 1

#ifdef __CUDACC__
            setBitCount = max(setBitCount, 1);
                queryImageUnsetBitCount = max(queryImageUnsetBitCount, 1);
#else
            setBitCount = std::max<unsigned int>(setBitCount, 1);
            queryImageUnsetBitCount = std::max<unsigned int>(queryImageUnsetBitCount, 1);
#endif

            // The fewer bits exist of a specific pixel type, the greater the penalty for not containing it
            float missedSetBitPenalty = float(totalBitsInBitString) / float(setBitCount);
            float missedUnsetBitPenalty = float(totalBitsInBitString) / float(queryImageUnsetBitCount);

            return {missedSetBitPenalty, missedUnsetBitPenalty};
        }

#ifdef __CUDACC__
        __host__
#endif
        inline HammingWeights computeWeightedHammingWeights(const ShapeDescriptor::QUICCIDescriptor &descriptor) {
            unsigned int setBitCount = 0;

            for(unsigned int i = 0; i < ShapeDescriptor::QUICCIDescriptorLength; i++) {
                setBitCount += std::bitset<32>(descriptor.contents[i]).count();
            }

            return computeWeightedHammingWeights(setBitCount, spinImageWidthPixels * spinImageWidthPixels);
        }

#ifdef __CUDACC__
        __device__
#endif
        inline HammingWeights computeWeightedHammingWeightsGPU(const ShapeDescriptor::QUICCIDescriptor &descriptor) {
            unsigned int setBitCount = 0;

            for(unsigned int i = 0; i < ShapeDescriptor::QUICCIDescriptorLength; i++) {
                setBitCount += __popc(descriptor.contents[i]);
            }

            return computeWeightedHammingWeights(setBitCount, spinImageWidthPixels * spinImageWidthPixels);
        }

#ifdef __CUDACC__
        __device__
#endif
        inline float computeChunkWeightedHammingDistance(const HammingWeights hammingWeights,
                                                         const unsigned int needle,
                                                         const unsigned int haystack) {
#ifdef __CUDACC__
            unsigned int missingSetPixelCount = __popc(needle & ~haystack);     // __popc((needle ^ haystack) & needle);
            unsigned int missingUnsetPixelCount = __popc(~needle & haystack);   // __popc((~needle ^ ~haystack) & ~needle);
#else
            unsigned int missingSetPixelCount = std::bitset<32>(needle & ~haystack).count();    // std::bitset<32>((needle ^ haystack) & needle).count();
            unsigned int missingUnsetPixelCount = std::bitset<32>(~needle & haystack).count();  // std::bitset<32>((~needle ^ ~haystack) & ~needle).count();
#endif

            return float(missingSetPixelCount) * hammingWeights.missingSetBitPenalty +
                   float(missingUnsetPixelCount) * hammingWeights.missingUnsetBitPenalty;
        }



#ifdef __CUDACC__
        __device__
#endif
        inline float computeWeightedHammingDistance(const HammingWeights hammingWeights,
                                                    const unsigned int* needle,
                                                    const unsigned int* haystack,
                                                    const unsigned int imageWidthBits,
                                                    const unsigned int imageHeightBits) {
            float distanceScore = 0;

            const unsigned int chunksPerRow = imageWidthBits / (8 * sizeof(unsigned int));

            unsigned int missingSetPixelCount = 0;
            unsigned int missingUnsetPixelCount = 0;

            for(unsigned int row = 0; row < imageHeightBits; row++) {
                for(unsigned int col = 0; col < chunksPerRow; col++) {
                    unsigned int needleChunk = needle[row * chunksPerRow + col];
                    unsigned int rightShiftedHaystackChunk = needleChunk >> 1;
                    unsigned int leftShiftedHaystackChunk = needleChunk << 1;

                    if(col > 0) {
                        unsigned int previousHaystackChunk = haystack[row * chunksPerRow + col - 1];
                        rightShiftedHaystackChunk = rightShiftedHaystackChunk | ((previousHaystackChunk & 0x1) << 31);
                    }
                    if(col < chunksPerRow - 1) {
                        unsigned int nextHaystackChunk = haystack[row * chunksPerRow + col + 1];
                        leftShiftedHaystackChunk = leftShiftedHaystackChunk | ((nextHaystackChunk >> 31) & 0x1);
                    }

                    unsigned int haystackChunk = haystack[row * chunksPerRow + col];
                    #ifdef __CUDACC__
                        missingSetPixelCount += __popc(needleChunk & ~haystackChunk); // (needle ^ haystack) & needle
                        missingUnsetPixelCount += __popc(~needleChunk & haystackChunk); // (~needle ^ ~haystack) & ~needle
                    #else
                        missingSetPixelCount += std::bitset<32>(needleChunk & ~haystackChunk).count();
                        missingUnsetPixelCount += std::bitset<32>(~needleChunk & haystackChunk).count();
                    #endif
/*
                    // We only need to penalise mismatched bits
                    unsigned int mismatchedBits = needleChunk ^ haystackChunk;

                    // if both neighbour bits are 0, no match exists. We thus use OR to combine the two directions
                    unsigned int shiftedMismatchedSetBits = needleChunk & ~(leftShiftedHaystackChunk | rightShiftedHaystackChunk);
                    // We want to penalise both neighbours being set to 1, because that means there's no matching 0 on either side.
                    unsigned int shiftedMismatchedUnsetBits = ~needleChunk & (leftShiftedHaystackChunk & rightShiftedHaystackChunk);

                    // Filter out bits we have already found matches for
                    shiftedMismatchedSetBits = shiftedMismatchedSetBits & mismatchedBits;
                    shiftedMismatchedUnsetBits = shiftedMismatchedUnsetBits & mismatchedBits;

                    #ifdef __CUDACC__
                        missingSetPixelCount += 2 * __popc(shiftedMismatchedSetBits); // (needle ^ haystack) & needle
                        missingUnsetPixelCount += 2 * __popc(shiftedMismatchedUnsetBits); // (~needle ^ ~haystack) & ~needle
                    #else
                        missingSetPixelCount += 2 * std::bitset<32>(shiftedMismatchedSetBits).count();
                        missingUnsetPixelCount += 2 * std::bitset<32>(shiftedMismatchedUnsetBits).count();
                    #endif
                */
                }
            }

            distanceScore = float(missingSetPixelCount) * hammingWeights.missingSetBitPenalty +
                            float(missingUnsetPixelCount) * hammingWeights.missingUnsetBitPenalty;

            return distanceScore;
        }
    }
}