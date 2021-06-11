#include "flexibleQUICCISearcher.h"

#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <chrono>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvidia/helper_cuda.h>
#include <cfloat>
#include <projectSymmetry/utilities/customWeightedHamming.cuh>


#ifndef warpSize
#define warpSize 32
#endif



const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;

__inline__ __device__ unsigned int warpAllReduceSum(unsigned int val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ unsigned int getChunkAt(const ShapeDescriptor::QUICCIDescriptor* image, const size_t imageIndex, const int chunkIndex) {
    return image[imageIndex].contents[chunkIndex];
}

__device__ int computeImageSumGPU(
        const ShapeDescriptor::QUICCIDescriptor* needleImages,
        const size_t imageIndex) {

    const int laneIndex = threadIdx.x % 32;

    unsigned int threadSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes images are multiples of warp size wide");

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int needleChunk = getChunkAt(needleImages, imageIndex, chunk);
        threadSum += __popc(needleChunk);
    }

    int sum = warpAllReduceSum(threadSum);

    return sum;
}

__device__ ShapeDescriptor::beta::distanceType compareConstantQUICCImagePairGPU(
        const ShapeDescriptor::QUICCIDescriptor* haystackImages,
        const size_t haystackImageIndex) {

    const int laneIndex = threadIdx.x % 32;

    ShapeDescriptor::beta::distanceType threadSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int haystackChunk =
                getChunkAt(haystackImages, haystackImageIndex, chunk);

        // Constant image is empty. Hence we only need to look at the haystack side of things.
#if QUICCI_DISTANCE == CLUTTER_RESISTANT_DISTANCE
        threadSum += __popc(haystackChunk);
#elif QUICCI_DISTANCE == HAMMING_DISTANCE
        threadSum += __popc(haystackChunk);
#elif QUICCI_DISTANCE == WEIGHTED_HAMMING_DISTANCE
        // Since a constant needle image will always use this function for ranking, and due to avoiding zero
        // division errors the weight of a missed unset bit is always 1, we can use the same ranking function
        // for weighted hamming as the other ranking functions.
        threadSum += float(__popc(haystackChunk));
#elif QUICCI_DISTANCE == FLEXIBLE_CLUTTER_RESISTANT
        threadSum += __popc(haystackChunk);
#endif
    }

    return warpAllReduceSum(threadSum);
}

__device__ ShapeDescriptor::beta::distanceType compareQUICCImagePairGPU(
        const ShapeDescriptor::QUICCIDescriptor* needleImages,
        const size_t needleImageIndex,
        const ShapeDescriptor::QUICCIDescriptor* haystackImages,
        const size_t haystackImageIndex
#if QUICCI_DISTANCE == WEIGHTED_HAMMING_DISTANCE
        , ProjectSymmetry::utilities::HammingWeights hammingWeights
#endif
) {


    const int laneIndex = threadIdx.x % 32;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    ShapeDescriptor::beta::distanceType threadScore = 0;

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int needleChunk = getChunkAt(needleImages, needleImageIndex, chunk);
        unsigned int haystackChunk = getChunkAt(haystackImages, haystackImageIndex, chunk);

#if QUICCI_DISTANCE == CLUTTER_RESISTANT_DISTANCE
        threadScore += __popc((needleChunk ^ haystackChunk) & needleChunk);
#elif QUICCI_DISTANCE == HAMMING_DISTANCE
        threadScore += __popc(needleChunk ^ haystackChunk);
#elif QUICCI_DISTANCE == WEIGHTED_HAMMING_DISTANCE
        threadScore += ProjectSymmetry::utilities::computeChunkWeightedHammingDistance(hammingWeights, needleChunk, haystackChunk);
#elif QUICCI_DISTANCE == FLEXIBLE_CLUTTER_RESISTANT
        static_assert(spinImageWidthPixels == 64, "This kernel assumes the image is 64x64 pixels");
        // Every pair of threads processes a row. There's a break point at bit 31 and 32 of the row,
        // so we need to exchange bits here to the other thread processing that row.
        const unsigned int isRightSide = (threadIdx.x & 1);
        const unsigned int isLeftSide = 1 - isRightSide;
        const unsigned int extraBit = (haystackChunk >> (isLeftSide * 31)) & 0x1;

        // First we start with accounting for exact bit matches
        // Since the distance function depends on bits in the query that are missing in the haystack, we'll compute those first.
        const unsigned int missingExactMatches = (needleChunk ^ haystackChunk) & needleChunk;

        // Discounting term for misses of the
        threadScore += __popc(missingExactMatches);

        // From the bits that did not have an exact match, we'll go look one pixel to the left and right
        unsigned int rightShifted = (haystackChunk >> 1) | ((isRightSide & extraBit) << 31);
        unsigned int leftShifted = (haystackChunk << 1) | (isLeftSide & extraBit);

        unsigned int missingLeftMatches = (missingExactMatches ^ rightShifted) & missingExactMatches;
        unsigned int missingRightMatches = (missingExactMatches ^ leftShifted) & missingExactMatches;

        unsigned int missingInBothDirections = missingLeftMatches ^ missingRightMatches;

        // Further discount if no matches are nearby
        threadScore += __popc(missingInBothDirections);
#endif
    }

    ShapeDescriptor::beta::distanceType imageScore = warpAllReduceSum(threadScore);

    return imageScore;
}


const unsigned int warpCount = 16;

__global__ void generateSearchResults(ShapeDescriptor::QUICCIDescriptor* needleDescriptors,
                                      size_t needleImageCount,
                                      ShapeDescriptor::QUICCIDescriptor* haystackDescriptors,
                                      size_t haystackImageCount,
                                      ShapeDescriptor::gpu::SearchResults<ShapeDescriptor::beta::distanceType>* searchResults) {

    size_t needleImageIndex = warpCount * blockIdx.x + (threadIdx.x / 32);

    if(needleImageIndex >= needleImageCount) {
        return;
    }

    static_assert(SEARCH_RESULT_COUNT == 128, "Array initialisation needs to change if search result count is changed");
    size_t threadSearchResultImageIndexes[SEARCH_RESULT_COUNT / 32] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
#if QUICCI_DISTANCE != WEIGHTED_HAMMING_DISTANCE
    ShapeDescriptor::beta::distanceType threadSearchResultScores[SEARCH_RESULT_COUNT / 32] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};
#else
    ShapeDescriptor::beta::distanceType threadSearchResultScores[SEARCH_RESULT_COUNT / 32] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
#endif
    const int blockCount = (SEARCH_RESULT_COUNT / 32);

#if QUICCI_DISTANCE == WEIGHTED_HAMMING_DISTANCE
    int referenceImageBitCount = computeImageSumGPU(needleDescriptors, needleImageIndex);
    ProjectSymmetry::utilities::HammingWeights hammingWeights = ProjectSymmetry::utilities::computeWeightedHammingWeights(referenceImageBitCount, spinImageWidthPixels * spinImageWidthPixels);
    #endif

    for(size_t haystackImageIndex = 0; haystackImageIndex < haystackImageCount; haystackImageIndex++) {
        ShapeDescriptor::beta::distanceType score = compareQUICCImagePairGPU(
                needleDescriptors,
                needleImageIndex,
                haystackDescriptors,
                haystackImageIndex
#if QUICCI_DISTANCE == WEIGHTED_HAMMING_DISTANCE
                , hammingWeights
#endif
        );

        // Since most images will not make it into the top ranking, we do a quick check to avoid a search
        // This saves a few instructions.
        if(score < __shfl_sync(0xFFFFFFFF, threadSearchResultScores[(SEARCH_RESULT_COUNT / 32) - 1], 31)) {
            unsigned int foundIndex = 0;
            for(int block = 0; block < blockCount; block++) {
                bool threadExceeds = threadSearchResultScores[block] > score;
                unsigned int bitString = __ballot_sync(0xFFFFFFFF, threadExceeds);
                unsigned int firstSet = __ffs(bitString) - 1;

                if(firstSet < 32) {
                    foundIndex = (block * 32) + (firstSet);
                    break;
                }
            }

            int startBlock = foundIndex / 32;
            const int endBlock = blockCount - 1;
            const int laneID = threadIdx.x % 32;

            // We first shift all values to the right for "full" 32-value blocks
            // Afterwards, we do one final iteration to shift only the values that are
            // block will never be 0, which ensures the loop body does not go out of range
            for(int block = endBlock; block > startBlock; block--) {
                int sourceThread = laneID - 1;
                int sourceBlock = block;

                if(laneID == 0) {
                    sourceThread = 31;
                }
                if(laneID == 31) {
                    sourceBlock = block - 1;
                }

                threadSearchResultScores[block] = __shfl_sync(0xFFFFFFFF, threadSearchResultScores[sourceBlock], sourceThread);
                threadSearchResultImageIndexes[block] = __shfl_sync(0xFFFFFFFF, threadSearchResultImageIndexes[sourceBlock], sourceThread);
            }

            // This shifts over values in the block where we're inserting the new value.
            // As such it requires some more fine-grained control.
            if(laneID >= foundIndex % 32) {
                int targetThread = laneID - 1;

                threadSearchResultScores[startBlock] = __shfl_sync(__activemask(), threadSearchResultScores[startBlock], targetThread);
                threadSearchResultImageIndexes[startBlock] = __shfl_sync(__activemask(), threadSearchResultImageIndexes[startBlock], targetThread);

                if(laneID == foundIndex % 32) {
                    threadSearchResultScores[startBlock] = score;
                    threadSearchResultImageIndexes[startBlock] = haystackImageIndex;
                }
            }

        }
    }


    const unsigned int laneID = threadIdx.x % 32;
    // Storing search results
    for(int block = 0; block < blockCount; block++) {
        searchResults[needleImageIndex].indices[block * 32 + laneID] = threadSearchResultImageIndexes[block];
        searchResults[needleImageIndex].scores[block * 32 + laneID] = threadSearchResultScores[block];
    }

}

ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<ShapeDescriptor::beta::distanceType>> ShapeDescriptor::beta::runFlexibleQUICCISearch(
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_needleDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_haystackDescriptors,
        float* executionTimeSeconds) {

    size_t searchResultBufferSize = device_needleDescriptors.length * sizeof(ShapeDescriptor::gpu::SearchResults<ShapeDescriptor::beta::distanceType>);
    ShapeDescriptor::gpu::SearchResults<ShapeDescriptor::beta::distanceType>* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));

    std::cout << "\t\tPerforming search.." << std::endl;
    auto start = std::chrono::steady_clock::now();

    generateSearchResults<<<(device_needleDescriptors.length / warpCount) + 1, 32 * warpCount>>>(
            device_needleDescriptors.content,
            device_needleDescriptors.length,
            device_haystackDescriptors.content,
            device_haystackDescriptors.length,
            device_searchResults);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    std::cout << "\t\t\tExecution time: " << duration.count() << std::endl;

    if(executionTimeSeconds != nullptr) {
        *executionTimeSeconds = float(duration.count()) / 1000.0f;
    }

    // Step 3: Copying results to CPU

    ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<ShapeDescriptor::beta::distanceType>> searchResults;
    searchResults.content = new ShapeDescriptor::gpu::SearchResults<ShapeDescriptor::beta::distanceType>[device_needleDescriptors.length];
    searchResults.length = device_needleDescriptors.length;

    checkCudaErrors(cudaMemcpy(searchResults.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

    // Cleanup

    cudaFree(device_searchResults);

    return searchResults;
}