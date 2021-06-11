#include "quicciStats.h"
#include <nvidia-samples-common/nvidia/helper_cuda.h>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include <numeric>
#include <projectSymmetry/types/Cluster.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>

const unsigned int imagesPerBlock = 64;

void computeOccurrenceCountsCPU(ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                                std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* outputCounts,
                                ShapeDescriptor::QUICCIDescriptor* ignoreMask) {
    std::fill(outputCounts->begin(), outputCounts->end(), 0);

    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

#pragma omp parallel for
    for(unsigned int imageIndex = 0; imageIndex < descriptors.length; imageIndex++) {
        for(unsigned int chunkIndex = 0; chunkIndex < UINTS_PER_QUICCI; chunkIndex++) {
            // Fetch chunk from memory
            unsigned int chunk = descriptors.content[imageIndex].contents[chunkIndex];

            // Apply ignore mask
            unsigned int chunkToIgnore = ignoreMask->contents[chunkIndex];
            chunk = chunk ^ chunkToIgnore;

            // Process chunk
            for(unsigned int bitInChunk = 0; bitInChunk < 32; bitInChunk++) {
                // To avoid bank conflicts and maintain good memory access patterns, we iterate over chunks within the warp
                unsigned int bitValue = (chunk >> (31 - bitInChunk)) & 0x1;

                if(bitValue == 1) {
                    unsigned int bitIndex = 8 * sizeof(unsigned int) * chunkIndex + bitInChunk;
#pragma omp atomic
                    outputCounts->at(bitIndex)++;
                }
            }
        }
    }
}


struct PixelReference {
    unsigned short pixelIndex = 0;
    unsigned int occurrenceCount = 0;
};


unsigned int computeOccurrenceLevel(
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
        std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels> *bitOrder,
        unsigned int imageIndex) {
    unsigned int levelReached = 0;

    for(unsigned int pixelIndexIndex = 0; pixelIndexIndex < spinImageWidthPixels * spinImageWidthPixels; pixelIndexIndex++) {
        unsigned short pixelIndex = bitOrder->at(pixelIndexIndex);
        unsigned int chunkIndex = pixelIndex / 32;
        unsigned int bitInChunkIndex = pixelIndex % 32;

        unsigned int chunk = descriptors.content[imageIndex].contents[chunkIndex];
        unsigned int bit = (chunk >> (31 - bitInChunkIndex)) & 0x1;

        if(bit == 1) {
            break;
        }
        levelReached++;
    }
    return std::min<unsigned int>(levelReached, spinImageWidthPixels * spinImageWidthPixels - 1);
}


void computeOccurrenceLevelsCPU(ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                                std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels> *bitOrder,
                                std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels> *outputLevels) {
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    for(unsigned int imageIndex = 0; imageIndex < descriptors.length; imageIndex++) {
        unsigned int levelReached = computeOccurrenceLevel(descriptors, bitOrder, imageIndex);

        for(unsigned int i = 0; i <= levelReached; i++) {
            outputLevels->at(i)++;
        }
    }

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);
}




void markImages(ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels> *bitOrder,
                unsigned int pivotLevel,
                std::vector<char> &imageHasReachedLevel) {
    for(unsigned int imageIndex = 0; imageIndex < descriptors.length; imageIndex++) {
        unsigned int levelReached = computeOccurrenceLevel(descriptors, bitOrder, imageIndex);

        if(levelReached >= pivotLevel) {
            imageHasReachedLevel.at(imageIndex) = 1;
        }
    }
}

void swapImages(ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                           ShapeDescriptor::cpu::array<ImageEntryMetadata> metadata,
                           std::vector<unsigned int> &targetIndices) {
    for(unsigned int imageIndex = 0; imageIndex < targetIndices.size(); imageIndex++) {
        unsigned int targetIndex = targetIndices.at(imageIndex);

        if (targetIndex != 0xFFFFFFFF) {
            for (unsigned int i = 0; i < UINTS_PER_QUICCI; i++) {
                unsigned int temp = descriptors.content[imageIndex].contents[i];
                descriptors.content[imageIndex].contents[i] = descriptors.content[targetIndex].contents[i];
                descriptors.content[targetIndex].contents[i] = temp;
            }

            ImageEntryMetadata tempMeta;
            tempMeta = metadata.content[imageIndex];
            metadata.content[imageIndex] = metadata.content[targetIndex];
            metadata.content[targetIndex] = tempMeta;
        }
    }
}

// TODO: make sure that it is guaranteed that some images end up on each side
// It may occur that only including all images in either side brings the total number over half of the images in the subset
void rearrangeImagesByLevelCPU(ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                               ShapeDescriptor::cpu::array<ImageEntryMetadata> metadata,
                               std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels>* bitOrder,
                               std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* levels,
                               unsigned int pivotLevel) {
    // First we'll run a kernel marking which images qualify for the set pivot level

    std::vector<char> imageHasReachedLevel(descriptors.length);
    std::fill(imageHasReachedLevel.begin(), imageHasReachedLevel.end(), 0);

    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    markImages(descriptors, bitOrder, pivotLevel, imageHasReachedLevel);

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    unsigned int matchingImageCount = levels->at(pivotLevel);
    std::vector<unsigned int> exchangeToImageIndex(matchingImageCount);
    std::fill(exchangeToImageIndex.begin(), exchangeToImageIndex.end(), 0xFFFFFFFF);

    unsigned int matchingBatchIndex = 0;
    unsigned int differingBatchIndex = matchingImageCount;

    while(matchingBatchIndex < matchingImageCount && differingBatchIndex < descriptors.length) {
        // Find next entry to swap in matching set by looking for image that does not reach level threshold
        while(imageHasReachedLevel[matchingBatchIndex] != 0 && matchingBatchIndex < matchingImageCount) {
            matchingBatchIndex++;
        }

        // Same thing for the differing batch, but the other way round
        while(imageHasReachedLevel[differingBatchIndex] != 1 && differingBatchIndex < descriptors.length) {
            differingBatchIndex++;
        }

        // At the end of the list, both loops will either have a valid pair, or ran out
        // Here we check for whether we did not run out, and therefore have a valid pair to swap
        if(matchingBatchIndex < matchingImageCount && differingBatchIndex < descriptors.length) {
            exchangeToImageIndex.at(matchingBatchIndex) = differingBatchIndex;

            // Moving on to the next pair
            matchingBatchIndex++;
            differingBatchIndex++;
        }
    }

    swapImages(descriptors, metadata, exchangeToImageIndex);
}




void computeNodeMaskImagesCPU(Cluster* tree) {
    for(unsigned int nodeIndex = 0; nodeIndex < tree->nodes.size(); nodeIndex++) {
        unsigned int startIndex = tree->nodes.at(nodeIndex).subtreeStartIndex;
        unsigned int endIndex = tree->nodes.at(nodeIndex).subtreeEndIndex;

        for(unsigned int i = 0; i < UINTS_PER_QUICCI; i++) {
            unsigned int productImageAccumulator = 0xFFFFFFFF;
            unsigned int sumImageAccumulator = 0x00000000;

            for (unsigned int imageIndex = startIndex; imageIndex < endIndex; imageIndex++) {
                productImageAccumulator = productImageAccumulator & tree->images.at(imageIndex).contents[i];
                sumImageAccumulator = sumImageAccumulator | tree->images.at(imageIndex).contents[i];
            }

            tree->nodes.at(nodeIndex).productImage.contents[i] = productImageAccumulator;
            tree->nodes.at(nodeIndex).sumImage.contents[i] = sumImageAccumulator;
        }
    }
}
