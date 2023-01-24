#include <nvidia/helper_cuda.h>
#include "binaryRICIConverter.h"

__device__ void writeQUICCImageHorizontal(
        ShapeDescriptor::QUICCIDescriptor* descriptorArray,
        radialIntersectionCountImagePixelType* RICIDescriptor,
        const int differenceThreshold) {

    const int laneIndex = threadIdx.x % 32;
    static_assert(spinImageWidthPixels % 32 == 0, "Implementation assumes images are multiples of warps wide");

    for(int row = 0; row < spinImageWidthPixels; row++) {
        radialIntersectionCountImagePixelType previousWarpLastNeedlePixelValue = 0;

        for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize) {
            radialIntersectionCountImagePixelType currentNeedlePixelValue =
                    RICIDescriptor[row * spinImageWidthPixels + pixel];

            int targetThread;
            if (laneIndex > 0) {
                targetThread = laneIndex - 1;
            }
            else if (pixel > 0) {
                targetThread = 31;
            } else {
                targetThread = 0;
            }

            radialIntersectionCountImagePixelType threadNeedleValue = 0;

            if (laneIndex == 31) {
                threadNeedleValue = previousWarpLastNeedlePixelValue;
            } else {
                threadNeedleValue = currentNeedlePixelValue;
            }

            radialIntersectionCountImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF, threadNeedleValue, targetThread);

            int imageDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);

            bool didIntersectionCountsChange = abs(imageDelta) >= differenceThreshold;

            unsigned int changeOccurredCombined = __brev(__ballot_sync(0xFFFFFFFF, didIntersectionCountsChange));

            if(laneIndex == 0) {
                size_t chunkIndex = (row * (spinImageWidthPixels / 32)) + (pixel / 32);
                descriptorArray[blockIdx.x].contents[chunkIndex] = changeOccurredCombined | descriptorArray[blockIdx.x].contents[chunkIndex];
            }

            // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
            previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
        }

    }
}

__device__ void writeQUICCImageVertical(
        ShapeDescriptor::QUICCIDescriptor* descriptorArray,
        radialIntersectionCountImagePixelType* RICIDescriptor,
        const int differenceThreshold) {

    const int laneIndex = threadIdx.x % 32;
    static_assert(spinImageWidthPixels % 32 == 0, "Implementation assumes images are multiples of warps wide");

    // col 0 turns into bottom row, should be leftmost col

    for(int col = 0; col < spinImageWidthPixels; col++) {
        radialIntersectionCountImagePixelType previousWarpLastNeedlePixelValue = 0;

        for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize) {
            radialIntersectionCountImagePixelType currentNeedlePixelValue =
                    RICIDescriptor[pixel * spinImageWidthPixels + col];

            int targetThread;
            if (laneIndex > 0) {
                targetThread = laneIndex - 1;
            }
            else if (pixel > 0) {
                targetThread = 31;
            } else {
                targetThread = 0;
            }

            radialIntersectionCountImagePixelType threadNeedleValue = 0;

            if (laneIndex == 31) {
                threadNeedleValue = previousWarpLastNeedlePixelValue;
            } else {
                threadNeedleValue = currentNeedlePixelValue;
            }

            radialIntersectionCountImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF, threadNeedleValue, targetThread);

            int imageDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);

            bool didIntersectionCountsChange = abs(imageDelta) >= differenceThreshold;

            unsigned int changeOccurredCombined = __brev(__ballot_sync(0xFFFFFFFF, didIntersectionCountsChange));

            if(laneIndex == 0) {
                size_t chunkIndex = (col * (spinImageWidthPixels / 32)) + (pixel / 32);
                descriptorArray[blockIdx.x].contents[chunkIndex] = changeOccurredCombined | descriptorArray[blockIdx.x].contents[chunkIndex];
            }

            // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
            previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
        }

    }

    __syncthreads();

    if (threadIdx.x == 0) {
    // transpose matrix

        int j, k;
        unsigned m, t;

        m = 0x0000FFFF;
        for (j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
            for (k = 0; k < 32; k = (k + j + 1) & ~j) {
                t = (descriptorArray[blockIdx.x].contents[k] ^ (descriptorArray[blockIdx.x].contents[k+j] >> j)) & m;
                descriptorArray[blockIdx.x].contents[k] = descriptorArray[blockIdx.x].contents[k] ^ t;
                descriptorArray[blockIdx.x].contents[k+j] = descriptorArray[blockIdx.x].contents[k+j] ^ (t << j);
            }
        }
    }
}

__global__ void doRICIConversion(ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> riciDescriptors,
                                 ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> convertedDescriptors,
                                 const int differenceThreshold) {
    const int laneIndex = threadIdx.x % 32;
#define imageIndex blockIdx.x
    static_assert(spinImageWidthPixels % 32 == 0);
    if (QUICCIDirection == 1) {
        writeQUICCImageHorizontal(convertedDescriptors.content, riciDescriptors.content[imageIndex].contents, differenceThreshold);
    }
    else if (QUICCIDirection == 2) {
        writeQUICCImageVertical(convertedDescriptors.content, riciDescriptors.content[imageIndex].contents, differenceThreshold);
    }
    else {
        writeQUICCImageVertical(convertedDescriptors.content, riciDescriptors.content[imageIndex].contents, differenceThreshold);
        writeQUICCImageHorizontal(convertedDescriptors.content, riciDescriptors.content[imageIndex].contents, differenceThreshold);
    }
    //     for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize) {
    //         radialIntersectionCountImagePixelType currentNeedlePixelValue =
    //                 riciDescriptors.content[imageIndex].contents[row * spinImageWidthPixels + pixel];

    //         int targetThread;
    //         if (laneIndex > 0) {
    //             targetThread = laneIndex - 1;
    //         }
    //         else if (pixel > 0) {
    //             targetThread = 31;
    //         } else {
    //             targetThread = 0;
    //         }

    //         radialIntersectionCountImagePixelType threadNeedleValue = 0;

    //         if (laneIndex == 31) {
    //             threadNeedleValue = previousWarpLastNeedlePixelValue;
    //         } else {
    //             threadNeedleValue = currentNeedlePixelValue;
    //         }

    //         radialIntersectionCountImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF, threadNeedleValue, targetThread);

    //         int imageDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);

    //         bool didIntersectionCountsChange = abs(imageDelta) >= differenceThreshold;

    //         unsigned int changeOccurredCombined = __brev(__ballot_sync(0xFFFFFFFF, didIntersectionCountsChange));

    //         if(laneIndex == 0) {
    //             size_t chunkIndex = (row * (spinImageWidthPixels / 32)) + (pixel / 32);
    //             convertedDescriptors.content[imageIndex].contents[chunkIndex] = changeOccurredCombined;
    //         }

    //         // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
    //         previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
    //     }

    // }
}

ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor>
convertRICIToModifiedQUICCI(ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors) {
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> convertedDescriptors(descriptors.length);

    doRICIConversion<<<descriptors.length, 32>>>(descriptors, convertedDescriptors, 2);
    

    checkCudaErrors(cudaDeviceSynchronize());

    return convertedDescriptors;
}
