#include <nvidia/helper_cuda.h>
#include "binaryRICIConverter.h"

__global__ void doRICIConversion(ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> riciDescriptors,
                                 ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> convertedDescriptors,
                                 const int differenceThreshold) {
    const int laneIndex = threadIdx.x % 32;
#define imageIndex blockIdx.x
    static_assert(spinImageWidthPixels % 32 == 0);

    for(int row = 0; row < spinImageWidthPixels; row++) {
        radialIntersectionCountImagePixelType previousWarpLastNeedlePixelValue = 0;

        for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize) {
            radialIntersectionCountImagePixelType currentNeedlePixelValue =
                    riciDescriptors.content[imageIndex].contents[row * spinImageWidthPixels + pixel];

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
                convertedDescriptors.content[imageIndex].contents[chunkIndex] = changeOccurredCombined;
            }

            // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
            previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
        }

    }
}

ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor>
convertRICIToModifiedQUICCI(ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors) {
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> convertedDescriptors(descriptors.length);

    doRICIConversion<<<descriptors.length, 32>>>(descriptors, convertedDescriptors, 2);

    checkCudaErrors(cudaDeviceSynchronize());

    return convertedDescriptors;
}
