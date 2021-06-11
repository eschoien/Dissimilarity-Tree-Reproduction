#include "ClusterBuilder.h"
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/read/QUICCIDescriptors.h>
#include <iostream>
#include "ClusteredQuiccImage.h"
#include "QuiccImageTreeNode.h"
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <projectSymmetry/descriptors/quicciStats.h>
#include <atomic>
#include <projectSymmetry/descriptors/quicciStatsCPU.h>


void indexImageBatch(Cluster* tree,
                     unsigned int* totalImagesIndexedThusFar,
                     ShapeDescriptor::gpu::array<ImageEntryMetadata> gpuImageMetadata,
                     ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> gpuDescriptors,
                     unsigned int minBatchSize,
                     ShapeDescriptor::QUICCIDescriptor ignoreMask) {

    for(unsigned int nodeIndex = 0; nodeIndex != tree->nodes.size(); nodeIndex++) {
        TreeNode currentNode = tree->nodes.at(nodeIndex);

        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorBatch;
        descriptorBatch.length = currentNode.subtreeEndIndex - currentNode.subtreeStartIndex;
        descriptorBatch.content = gpuDescriptors.content + currentNode.subtreeStartIndex;

        ShapeDescriptor::gpu::array<ImageEntryMetadata> metadataBatch;
        metadataBatch.length = currentNode.subtreeEndIndex - currentNode.subtreeStartIndex;
        metadataBatch.content = gpuImageMetadata.content + currentNode.subtreeStartIndex;

        // Stop subdividing the tree when batches are small enough
        if (descriptorBatch.length <= minBatchSize) {
            // Marking node as leaf node
            tree->nodes.at(nodeIndex).matchingNodeID = 0xFFFFFFFF;
            tree->nodes.at(nodeIndex).differingNodeID = 0xFFFFFFFF;
            *totalImagesIndexedThusFar += descriptorBatch.length;
            continue;
        }

        //std::cout << "Processing node " << nodeIndex << " with " << descriptorBatch.length << " images" << std::endl;

        std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels> occurrenceCounts;
        std::fill(occurrenceCounts.begin(), occurrenceCounts.end(), 0);
        computeOccurrenceCounts(descriptorBatch, &occurrenceCounts, &ignoreMask);

        std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels> pixelOrder = computePixelOrder(
                &occurrenceCounts);

        std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels> levels;
        std::fill(levels.begin(), levels.end(), 0);
        computeOccurrenceLevels(descriptorBatch, &pixelOrder, &levels);

        unsigned int pivotLevel = computePivotLevel(&levels, descriptorBatch.length);

        unsigned int matchingImageCount = 0;
        if(pivotLevel < spinImageWidthPixels * spinImageWidthPixels) {
            matchingImageCount = levels.at(pivotLevel);
        }

        // Images cannot be divided further. Most likely they are equivalent, so we mark this as a leaf node
        if(pivotLevel == spinImageWidthPixels * spinImageWidthPixels
           || matchingImageCount == descriptorBatch.length
           || matchingImageCount == 0) {
            // Marking node as leaf node
            tree->nodes.at(nodeIndex).matchingNodeID = 0xFFFFFFFF;
            tree->nodes.at(nodeIndex).differingNodeID = 0xFFFFFFFF;
            *totalImagesIndexedThusFar += descriptorBatch.length;
            continue;
        }

        //std::cout << "Chose level " << pivotLevel << " as pivot with " << matchingImageCount << " matches." << std::endl;

        rearrangeImagesByLevel(descriptorBatch, metadataBatch, &pixelOrder, &levels, pivotLevel);

        // Apply mask

        ShapeDescriptor::QUICCIDescriptor matchingIgnoreMask = ignoreMask;
        for (unsigned int i = 0; i < pivotLevel; i++) {
            unsigned int chunkIndex = pixelOrder.at(i) / 32;
            unsigned int bitIndex = pixelOrder.at(i) % 32;

            matchingIgnoreMask.contents[chunkIndex] =
                    matchingIgnoreMask.contents[chunkIndex] | (0x1U << (31U - bitIndex));
        }

        // Create new tree nodes

        TreeNode matchingNode;
        matchingNode.subtreeStartIndex = currentNode.subtreeStartIndex;
        matchingNode.subtreeEndIndex = currentNode.subtreeStartIndex + matchingImageCount;
        matchingNode.sumImage = matchingIgnoreMask;
        tree->nodes.push_back(matchingNode);
        unsigned int matchingNodeID = tree->nodes.size() - 1;
        tree->nodes.at(nodeIndex).matchingNodeID = matchingNodeID;

        TreeNode differingNode;
        differingNode.subtreeStartIndex = currentNode.subtreeStartIndex + matchingImageCount;
        differingNode.subtreeEndIndex = currentNode.subtreeEndIndex;
        tree->nodes.push_back(differingNode);
        unsigned int differingNodeID = tree->nodes.size() - 1;
        tree->nodes.at(nodeIndex).differingNodeID = differingNodeID;

        std::stringstream out;
        out << nodeIndex << " - " << tree->nodes.size() << " -> " << descriptorBatch.length << "\tindexed " << *totalImagesIndexedThusFar << "/" << gpuDescriptors.length << std::endl;
        std::cout << out.str();
    }
}


void indexImageBatchCPU(Cluster* tree,
                     unsigned int* totalImagesIndexedThusFar,
                     unsigned int minBatchSize,
                     ShapeDescriptor::QUICCIDescriptor ignoreMask) {

    for(unsigned int nodeIndex = 0; nodeIndex != tree->nodes.size(); nodeIndex++) {
        TreeNode currentNode = tree->nodes.at(nodeIndex);

        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorBatch;
        descriptorBatch.length = currentNode.subtreeEndIndex - currentNode.subtreeStartIndex;
        descriptorBatch.content = tree->images.data() + currentNode.subtreeStartIndex;

        ShapeDescriptor::cpu::array<ImageEntryMetadata> metadataBatch;
        metadataBatch.length = currentNode.subtreeEndIndex - currentNode.subtreeStartIndex;
        metadataBatch.content = tree->imageMetadata.data() + currentNode.subtreeStartIndex;

        // Stop subdividing the tree when batches are small enough
        if (descriptorBatch.length <= minBatchSize) {
            // Marking node as leaf node
            tree->nodes.at(nodeIndex).matchingNodeID = 0xFFFFFFFF;
            tree->nodes.at(nodeIndex).differingNodeID = 0xFFFFFFFF;
            *totalImagesIndexedThusFar += descriptorBatch.length;
            continue;
        }

        //std::cout << "Processing node " << nodeIndex << " with " << descriptorBatch.length << " images" << std::endl;

        std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels> occurrenceCounts;
        std::fill(occurrenceCounts.begin(), occurrenceCounts.end(), 0);
        computeOccurrenceCountsCPU(descriptorBatch, &occurrenceCounts, &ignoreMask);

        std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels> pixelOrder = computePixelOrder(
                &occurrenceCounts);

        std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels> levels;
        std::fill(levels.begin(), levels.end(), 0);
        computeOccurrenceLevelsCPU(descriptorBatch, &pixelOrder, &levels);

        unsigned int pivotLevel = computePivotLevel(&levels, descriptorBatch.length);

        unsigned int matchingImageCount = 0;
        if(pivotLevel < spinImageWidthPixels * spinImageWidthPixels) {
            matchingImageCount = levels.at(pivotLevel);
        }

        // Images cannot be divided further. Most likely they are equivalent, so we mark this as a leaf node
        if(pivotLevel == spinImageWidthPixels * spinImageWidthPixels
           || matchingImageCount == descriptorBatch.length
           || matchingImageCount == 0) {
            // Marking node as leaf node
            tree->nodes.at(nodeIndex).matchingNodeID = 0xFFFFFFFF;
            tree->nodes.at(nodeIndex).differingNodeID = 0xFFFFFFFF;
            *totalImagesIndexedThusFar += descriptorBatch.length;
            continue;
        }

        //std::cout << "Chose level " << pivotLevel << " as pivot with " << matchingImageCount << " matches." << std::endl;

        rearrangeImagesByLevelCPU(descriptorBatch, metadataBatch, &pixelOrder, &levels, pivotLevel);

        // Apply mask

        ShapeDescriptor::QUICCIDescriptor matchingIgnoreMask = ignoreMask;
        for (unsigned int i = 0; i < pivotLevel; i++) {
            unsigned int chunkIndex = pixelOrder.at(i) / 32;
            unsigned int bitIndex = pixelOrder.at(i) % 32;

            matchingIgnoreMask.contents[chunkIndex] =
                    matchingIgnoreMask.contents[chunkIndex] | (0x1U << (31U - bitIndex));
        }

        // Create new tree nodes

        TreeNode matchingNode;
        matchingNode.subtreeStartIndex = currentNode.subtreeStartIndex;
        matchingNode.subtreeEndIndex = currentNode.subtreeStartIndex + matchingImageCount;
        matchingNode.sumImage = matchingIgnoreMask;
        tree->nodes.push_back(matchingNode);
        unsigned int matchingNodeID = tree->nodes.size() - 1;
        tree->nodes.at(nodeIndex).matchingNodeID = matchingNodeID;

        TreeNode differingNode;
        differingNode.subtreeStartIndex = currentNode.subtreeStartIndex + matchingImageCount;
        differingNode.subtreeEndIndex = currentNode.subtreeEndIndex;
        tree->nodes.push_back(differingNode);
        unsigned int differingNodeID = tree->nodes.size() - 1;
        tree->nodes.at(nodeIndex).differingNodeID = differingNodeID;

        std::stringstream out;
        out << nodeIndex << " - " << tree->nodes.size() << " -> " << descriptorBatch.length << "\tindexed " << *totalImagesIndexedThusFar << "/" << tree->images.size() << std::endl;
        std::cout << out.str();
    }
}

Cluster* buildClusterFromDumpDirectory(const cluster::path &imageDumpDirectory,
                                       const cluster::path &indexDirectory,
                                       const unsigned int imagesPerBucket,
                                       bool forceCPU) {
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(imageDumpDirectory);

    std::cout << "Counting images to index.." << std::endl;
    size_t imageCountToIndex = 0;
    #pragma omp parallel for schedule(dynamic)
    for(unsigned int i = 0; i < haystackFiles.size(); i++) {
        ShapeDescriptor::QUICCIDescriptorFileHeader header = ShapeDescriptor::read::QuicciDescriptorFileHeader(haystackFiles.at(i));
        #pragma omp atomic
        imageCountToIndex += header.imageCount;
    }
    std::cout << "Found " << imageCountToIndex << " images in directory" << std::endl;

    std::atomic<unsigned int> nextStartIndex;
    nextStartIndex = 0;

    Cluster* cluster = new Cluster;
    cluster->images.resize(imageCountToIndex);
    cluster->imageMetadata.resize(imageCountToIndex);
    cluster->maxImagesPerLeafNode = imagesPerBucket;
    cluster->indexedFiles = haystackFiles;

    std::cout << "Loading descriptors.." << std::endl;

    #pragma omp parallel for schedule(dynamic)
    for(unsigned int i = 0; i < haystackFiles.size(); i++) {
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::read::QUICCIDescriptors(haystackFiles.at(i));
        unsigned int startIndex = nextStartIndex.fetch_add(descriptors.length, std::memory_order_relaxed);
        std::copy(descriptors.content, descriptors.content + descriptors.length, &cluster->images[startIndex]);
        for(unsigned int imageIndex = startIndex; imageIndex < startIndex + descriptors.length; imageIndex++) {
            cluster->imageMetadata.at(imageIndex).imageID = imageIndex - startIndex;
            cluster->imageMetadata.at(imageIndex).fileID = i;
        }
        ShapeDescriptor::free::array(descriptors);
    }


    std::cout << "Constructing index tree.." << std::endl;

    TreeNode rootNode;
    rootNode.subtreeStartIndex = 0;
    rootNode.subtreeEndIndex = imageCountToIndex;
    std::fill(rootNode.sumImage.contents, rootNode.sumImage.contents + UINTS_PER_QUICCI, 0);
    std::fill(rootNode.productImage.contents, rootNode.productImage.contents + UINTS_PER_QUICCI, 0);
    cluster->nodes.push_back(rootNode);
    ShapeDescriptor::QUICCIDescriptor ignoreMask;
    std::fill(ignoreMask.contents, ignoreMask.contents + UINTS_PER_QUICCI, 0);
    unsigned int totalImagesIndexedThusFar = 0;

    if(!forceCPU) {
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> gpuDescriptors(imageCountToIndex);
        checkCudaErrors(cudaMemcpy(gpuDescriptors.content, cluster->images.data(), sizeof(ShapeDescriptor::QUICCIDescriptor) * gpuDescriptors.length, cudaMemcpyHostToDevice));
        ShapeDescriptor::gpu::array<ImageEntryMetadata> gpuImageMetadata(imageCountToIndex);
        checkCudaErrors(cudaMemcpy(gpuImageMetadata.content, cluster->imageMetadata.data(), sizeof(ImageEntryMetadata) * gpuImageMetadata.length, cudaMemcpyHostToDevice));

        indexImageBatch(cluster, &totalImagesIndexedThusFar, gpuImageMetadata, gpuDescriptors, imagesPerBucket, ignoreMask);

        std::cout << "Computing branch images.." << std::endl;

        computeNodeMaskImages(gpuDescriptors, cluster);

        std::cout << "Copying rearranged descriptors back into RAM.." << std::endl;

        checkCudaErrors(cudaMemcpy(cluster->images.data(), gpuDescriptors.content, sizeof(ShapeDescriptor::QUICCIDescriptor) * gpuDescriptors.length, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(cluster->imageMetadata.data(), gpuImageMetadata.content, sizeof(ImageEntryMetadata) * gpuImageMetadata.length, cudaMemcpyDeviceToHost));

        ShapeDescriptor::free::array(gpuDescriptors);
        ShapeDescriptor::free::array(gpuImageMetadata);
    } else {
        std::cout << "Indexing images on the CPU.." << std::endl;
        std::cout << "Hope you have a cup of coffee at hand." << std::endl;
        std::cout << "If not, this is an excellent time to go fectch one" << std::endl;

        indexImageBatchCPU(cluster, &totalImagesIndexedThusFar, imagesPerBucket, ignoreMask);

        std::cout << "Computing branch images.." << std::endl;

        computeNodeMaskImagesCPU(cluster);
    }

    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl << "Index construction complete. " << std::endl;
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    return cluster;
}
