//
// Created by bart on 07.01.2021.
//

#include "RICIUtils.h"


ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> computeRICIDescriptorsBatched(
        ShapeDescriptor::gpu::Mesh mesh,
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
        float supportRadius,
        size_t imagesPerBatch = 10000,
        bool showProgress = false) {

    // The array which will contain the complete set of rendered descriptors
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors(descriptorOrigins.length);

    // For each batch
    for(size_t startIndex = 0; startIndex < descriptorOrigins.length; startIndex += imagesPerBatch) {
        if(showProgress) {
            std::cout << "\r\tRendering images.. (Processing " << startIndex << "/" << descriptorOrigins.length << ")" << std::flush;
        }

        // Compute how many images we need to render, and create an array of descriptor origins that represents
        // the current batch of images we want to render
        size_t imageCountInBatch = std::min<size_t>(imagesPerBatch, descriptorOrigins.length - startIndex);
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOriginBatch;
        descriptorOriginBatch.content = descriptorOrigins.content + startIndex;
        descriptorOriginBatch.length = imageCountInBatch;

        // Actually render the descriptors
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptorBatch = ShapeDescriptor::gpu::generateRadialIntersectionCountImages(mesh, descriptorOriginBatch, supportRadius);

        // Copy the rendered descriptors into the large output array
        checkCudaErrors(cudaMemcpy(
                descriptors.content + startIndex,
                descriptorBatch.content,
                imageCountInBatch * sizeof(ShapeDescriptor::OrientedPoint),
                cudaMemcpyDeviceToDevice));

        // Free the small batch of rendered images
        ShapeDescriptor::free::array(descriptorBatch);
    }

    if(showProgress) {
        std::cout << std::endl;
    }

    return descriptors;
}


ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<unsigned int>> computeRICISearchResultsBatched(
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> needleDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> haystackDescriptors,
        size_t batchSize = 10000,
        bool showProgress = false) {

    // Allocate the array which will contain the combined search results
    ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<unsigned int>> searchResults(needleDescriptors.length);

    // For each batch
    for(size_t startIndex = 0; startIndex < needleDescriptors.length; startIndex += batchSize) {
        if(showProgress) {
            std::cout << "\r\tComputing search results.. (Processing " << startIndex << "/" << needleDescriptors.length << ")" << std::flush;
        }

        size_t batchLength = std::min<size_t>(batchSize, needleDescriptors.length - startIndex);

        // Create a dummy array which represents the batch of images to compute search results for
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> batchNeedleDescriptors;
        batchNeedleDescriptors.content = needleDescriptors.content + startIndex;
        batchNeedleDescriptors.length = batchLength;

        // Compute search results for this batch of descriptors
        ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<unsigned int>> batchSearchResults = ShapeDescriptor::gpu::findRadialIntersectionCountImagesInHaystack(batchNeedleDescriptors, haystackDescriptors);

        // Copy search results into array of complete search results
        std::copy(batchSearchResults.content, batchSearchResults.content + batchLength, searchResults.content);

        // Free the batch of search results
        ShapeDescriptor::free::array(batchSearchResults);
    }

    if(showProgress) {
        std::cout << std::endl;
    }

    return searchResults;
}