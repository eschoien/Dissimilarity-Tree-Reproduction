#include "quicciStats.h"
#include <nvidia-samples-common/nvidia/helper_cuda.h>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include <numeric>
#include <projectSymmetry/types/Cluster.h>

const unsigned int imagesPerBlock = 64;

__global__ void countOccurrences(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                                 unsigned int* outputCounts,
                                 ShapeDescriptor::QUICCIDescriptor* ignoreMask) {
    unsigned int startImageIndex = blockIdx.x * imagesPerBlock;
    unsigned int endImageIndex = min((unsigned int) ((blockIdx.x + 1) * imagesPerBlock),
                                     (unsigned int) descriptors.length);

    __shared__ unsigned int localCounts[spinImageWidthPixels * spinImageWidthPixels];
    for(unsigned int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        localCounts[i] = 0;
    }

    __syncthreads();

    const unsigned int laneIndex = threadIdx.x % 32;
    const unsigned int warpIndex = threadIdx.x / 32;
    const unsigned int warpCount = blockDim.x / 32;

    for(unsigned int imageIndex = startImageIndex; imageIndex < endImageIndex; imageIndex++) {
        for(unsigned int chunkBatchIndex = warpIndex; chunkBatchIndex < UINTS_PER_QUICCI / 32; chunkBatchIndex += warpCount) {
            unsigned int chunkIndex = 32 * chunkBatchIndex + laneIndex;

            // Fetch chunk from memory
            unsigned int chunk = descriptors.content[imageIndex].contents[chunkIndex];

            // Apply ignore mask
            unsigned int chunkToIgnore = ignoreMask->contents[chunkIndex];
            chunk = chunk ^ chunkToIgnore;

            // Process chunk
            for(unsigned int chunkInBatch = 0; chunkInBatch < 32; chunkInBatch++) {
                // To avoid bank conflicts and maintain good memory access patterns, we iterate over chunks within the warp
                unsigned int currentChunk = __shfl_sync(0xFFFFFFFF, chunk, chunkInBatch);
                unsigned int bitInChunk = (currentChunk >> (31 - laneIndex)) & 0x1;
                unsigned int bitIndexInImage = 32 * 32 * chunkBatchIndex + 32 * chunkInBatch + laneIndex;

                if(bitInChunk == 1) {
                    atomicAdd(&localCounts[bitIndexInImage], 1);
                }
            }
        }
    }

    __syncthreads();

    for(unsigned int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        atomicAdd(&outputCounts[i], localCounts[i]);
    }
}

void computeOccurrenceCounts(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                             std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* outputCounts,
                             ShapeDescriptor::QUICCIDescriptor* ignoreMask) {

    unsigned int* device_outputCounts;
    const unsigned int outputBufferSize = sizeof(unsigned int) * spinImageWidthPixels * spinImageWidthPixels;
    checkCudaErrors(cudaMalloc(&device_outputCounts, outputBufferSize));
    checkCudaErrors(cudaMemset(device_outputCounts, 0, outputBufferSize));

    ShapeDescriptor::QUICCIDescriptor* device_ignoreMask;
    checkCudaErrors(cudaMalloc(&device_ignoreMask, sizeof(ShapeDescriptor::QUICCIDescriptor)));
    checkCudaErrors(cudaMemcpy(device_ignoreMask, ignoreMask, sizeof(ShapeDescriptor::QUICCIDescriptor), cudaMemcpyHostToDevice));

    unsigned int blockCount = descriptors.length / imagesPerBlock;
    if(descriptors.length % imagesPerBlock != 0) {
        blockCount++;
    }

    unsigned int threadsPerBlock = 128;

    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    countOccurrences<<<blockCount, threadsPerBlock>>>(descriptors, device_outputCounts, device_ignoreMask);

    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);
    //std::cout << "Execution time: " << totalExecutionDuration.count() << "ms" << std::endl;

    checkCudaErrors(cudaMemcpy(outputCounts->data(), device_outputCounts, outputBufferSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(device_outputCounts));
    checkCudaErrors(cudaFree(device_ignoreMask));
}







struct PixelReference {
    unsigned short pixelIndex = 0;
    unsigned int occurrenceCount = 0;
};

bool pixelReferenceComparator(const PixelReference& left, const PixelReference& right)
{
    // We want to sort occurrences in descending order
    return left.occurrenceCount < right.occurrenceCount;
}

std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels>
computePixelOrder(std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels> *outputCounts) {
    std::array<PixelReference, spinImageWidthPixels * spinImageWidthPixels> references;
    for(unsigned int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
        references.at(i).pixelIndex = i;
        references.at(i).occurrenceCount = outputCounts->at(i);
    }

    std::sort(references.begin(), references.end(), pixelReferenceComparator);

    std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels> pixelIndices;
    for(unsigned int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
        pixelIndices.at(i) = references.at(i).pixelIndex;

        //std::cout << "(" << pixelIndices.at(i) << ", " << references.at(i).occurrenceCount << "), ";
        //if(i % 64 == 63) std::cout << std::endl;
    }

    return pixelIndices;
}





// TODO: start at first nonzero occurrence in bit order (should not be necessary, but might speed up some computation)
__device__ __inline__ unsigned int computeOccurrenceLevel(
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
        unsigned short* bitOrder,
        unsigned int imageIndex) {
    unsigned int levelReached = 0;

    for(unsigned int pixelIndexIndex = threadIdx.x; pixelIndexIndex < spinImageWidthPixels * spinImageWidthPixels; pixelIndexIndex += blockDim.x) {
        unsigned short pixelIndex = bitOrder[pixelIndexIndex];
        unsigned int chunkIndex = pixelIndex / 32;
        unsigned int bitInChunkIndex = pixelIndex % 32;

        // Dear memory system, I hereby preemptively apologise
        unsigned int chunk = descriptors.content[imageIndex].contents[chunkIndex];
        unsigned int bit = (chunk >> (31 - bitInChunkIndex)) & 0x1;

        unsigned int warpDetections = __ballot_sync(0xFFFFFFFF, bit == 0);
        if(warpDetections != 0xFFFFFFFF) {
            // Inefficient, but this kernel runs fast enough anyway, so who cares.
            while((warpDetections & 0x1) == 0x1) {
                levelReached += 1;
                warpDetections = warpDetections >> 1;
            }
            return levelReached;
        } else {
            levelReached += 32;
        }
    }
    return levelReached;
}

__global__ void countOccurrenceLevels(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                                      unsigned short* bitOrder,
                                      unsigned int* outputLevels) {
    const unsigned int imageIndex = blockIdx.x;

    unsigned int levelReached = computeOccurrenceLevel(descriptors, bitOrder, imageIndex);

    for(unsigned int i = threadIdx.x; i <= levelReached; i += blockDim.x) {
        atomicAdd(&outputLevels[i], 1);
    }
}

void computeOccurrenceLevels(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                             std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels> *bitOrder,
                             std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels> *outputLevels) {
    unsigned int* device_outputLevels;
    const unsigned int outputBufferSize = sizeof(unsigned int) * spinImageWidthPixels * spinImageWidthPixels;
    checkCudaErrors(cudaMalloc(&device_outputLevels, outputBufferSize));
    checkCudaErrors(cudaMemset(device_outputLevels, 0, outputBufferSize));

    unsigned short* device_bitOrder;
    const unsigned int bitOrderBufferSize = sizeof(unsigned short) * spinImageWidthPixels * spinImageWidthPixels;
    checkCudaErrors(cudaMalloc(&device_bitOrder, bitOrderBufferSize));
    checkCudaErrors(cudaMemcpy(device_bitOrder, bitOrder, bitOrderBufferSize, cudaMemcpyHostToDevice));

    unsigned int threadsPerBlock = 32;

    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    countOccurrenceLevels<<<descriptors.length, threadsPerBlock>>>(descriptors, device_bitOrder, device_outputLevels);

    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);
    //std::cout << "Execution time: " << totalExecutionDuration.count() << "ms" << std::endl;

    checkCudaErrors(cudaMemcpy(outputLevels->data(), device_outputLevels, outputBufferSize, cudaMemcpyDeviceToHost));

    for(unsigned int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
        if(outputLevels->at(i) == 0) {
            continue;
        }
        //std::cout << "\t" << i << ": " << outputLevels->at(i) << " - score: " << (i * outputLevels->at(i)) << std::endl;
    }

    checkCudaErrors(cudaFree(device_outputLevels));
    checkCudaErrors(cudaFree(device_bitOrder));
}







unsigned int computePivotLevel(std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* levels, unsigned int batchImageCount) {
    const unsigned int highestPossibleLevel = spinImageWidthPixels * spinImageWidthPixels - 1;

    unsigned int halfOfImages = batchImageCount / 2;
    for(unsigned int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
        size_t imageCountAtLevel = ((size_t) levels->at(i));
        if(imageCountAtLevel < halfOfImages) {
            return std::min(highestPossibleLevel - 1, i);
        }
    }
    return spinImageWidthPixels * spinImageWidthPixels;
}








__global__ void markImages(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                                      unsigned short* bitOrder,
                                      unsigned int pivotLevel,
                                      char* imageHasReachedLevel) {
    const unsigned int imageIndex = blockIdx.x;

    unsigned int levelReached = computeOccurrenceLevel(descriptors, bitOrder, imageIndex);

    if(threadIdx.x == 0 && levelReached >= pivotLevel) {
        imageHasReachedLevel[imageIndex] = 1;
    }
}

__global__ void swapImages(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                           ShapeDescriptor::gpu::array<ImageEntryMetadata> metadata,
                           const unsigned int* targetIndices) {
    const unsigned int imageIndex = blockIdx.x;

    unsigned int targetIndex = targetIndices[imageIndex];

    if(targetIndex != 0xFFFFFFFF) {
        for(unsigned int i = threadIdx.x; i < UINTS_PER_QUICCI; i += blockDim.x) {
            unsigned int temp = descriptors.content[imageIndex].contents[i];
            descriptors.content[imageIndex].contents[i] = descriptors.content[targetIndex].contents[i];
            descriptors.content[targetIndex].contents[i] = temp;
        }

        if(threadIdx.x == 0) {
            ImageEntryMetadata tempMeta;
            tempMeta = metadata.content[imageIndex];
            metadata.content[imageIndex] = metadata.content[targetIndex];
            metadata.content[targetIndex] = tempMeta;
        }
    }
}

// TODO: make sure that it is guaranteed that some images end up on each side
// It may occur that only including all images in either side brings the total number over half of the images in the subset
void rearrangeImagesByLevel(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                            ShapeDescriptor::gpu::array<ImageEntryMetadata> metadata,
                                    std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels>* bitOrder,
                                    std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* levels,
                                    unsigned int pivotLevel) {
    // First we'll run a kernel marking which images qualify for the set pivot level

    char* device_imageHasReachedLevel;
    const unsigned int markedImageBufferSize = sizeof(char) * descriptors.length;
    checkCudaErrors(cudaMalloc(&device_imageHasReachedLevel, markedImageBufferSize));
    checkCudaErrors(cudaMemset(device_imageHasReachedLevel, 0, markedImageBufferSize));

    unsigned short* device_bitOrder;
    const unsigned int bitOrderBufferSize = sizeof(unsigned short) * spinImageWidthPixels * spinImageWidthPixels;
    checkCudaErrors(cudaMalloc(&device_bitOrder, bitOrderBufferSize));
    checkCudaErrors(cudaMemcpy(device_bitOrder, bitOrder, bitOrderBufferSize, cudaMemcpyHostToDevice));

    unsigned int threadsPerBlock = 32;

    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    markImages<<<descriptors.length, threadsPerBlock>>>(descriptors, device_bitOrder, pivotLevel, device_imageHasReachedLevel);

    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);
    //std::cout << "Execution time: " << totalExecutionDuration.count() << "ms" << std::endl;

    // We'll do a quick counting session on the CPU

    unsigned int matchingImageCount = levels->at(pivotLevel);

    std::vector<char> imageHasReachedLevel(descriptors.length);
    std::vector<unsigned int> exchangeToImageIndex(matchingImageCount);
    std::fill(exchangeToImageIndex.begin(), exchangeToImageIndex.end(), 0xFFFFFFFF);
    checkCudaErrors(cudaMemcpy(imageHasReachedLevel.data(), device_imageHasReachedLevel, markedImageBufferSize, cudaMemcpyDeviceToHost));

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

    unsigned int* device_targetIndices;
    const size_t targetIndicesBufferSize = sizeof(unsigned int) * matchingImageCount;
    checkCudaErrors(cudaMalloc(&device_targetIndices, targetIndicesBufferSize));
    checkCudaErrors(cudaMemcpy(device_targetIndices, exchangeToImageIndex.data(), targetIndicesBufferSize, cudaMemcpyHostToDevice));

    swapImages<<<matchingImageCount, 32>>>(descriptors, metadata, device_targetIndices);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(device_targetIndices));
    checkCudaErrors(cudaFree(device_imageHasReachedLevel));
    checkCudaErrors(cudaFree(device_bitOrder));
}














__global__ void computeNodeMaskImage(TreeNode* treeNodes,
                                     ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors) {
    unsigned int nodeIndex = blockIdx.x;
    unsigned int startIndex = treeNodes[nodeIndex].subtreeStartIndex;
    unsigned int endIndex = treeNodes[nodeIndex].subtreeEndIndex;

    for(unsigned int i = threadIdx.x; i < UINTS_PER_QUICCI; i += blockDim.x) {
        unsigned int productImageAccumulator = 0xFFFFFFFF;
        unsigned int sumImageAccumulator = 0x00000000;

        for (unsigned int imageIndex = startIndex; imageIndex < endIndex; imageIndex++) {
            productImageAccumulator = productImageAccumulator & descriptors.content[imageIndex].contents[i];
            sumImageAccumulator = sumImageAccumulator | descriptors.content[imageIndex].contents[i];
        }

        treeNodes[nodeIndex].productImage.contents[i] = productImageAccumulator;
        treeNodes[nodeIndex].sumImage.contents[i] = sumImageAccumulator;
    }
}

void computeNodeMaskImages(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                           Cluster* tree) {
    TreeNode* device_treeNodes;
    size_t treeNodeBufferSize = tree->nodes.size() * sizeof(TreeNode);
    checkCudaErrors(cudaMalloc(&device_treeNodes, treeNodeBufferSize));
    checkCudaErrors(cudaMemcpy(device_treeNodes, tree->nodes.data(), treeNodeBufferSize, cudaMemcpyHostToDevice));

    computeNodeMaskImage<<<tree->nodes.size(), 128>>>(device_treeNodes, descriptors);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(tree->nodes.data(), device_treeNodes, treeNodeBufferSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(device_treeNodes));
}