#include <iostream>
#include <arrrgh.hpp>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <json.hpp>
#include <json/tsl/ordered_map.h>
#include <git.h>
#include <projectSymmetry/descriptors/binaryRICIConverter.h>
#include <projectSymmetry/descriptors/flexibleQUICCISearcher.h>
#include <projectSymmetry/clustering/IndexQueryer.h>
#include <projectSymmetry/clustering/ClusterIO.h>
#include <shapeDescriptor/utilities/dump/descriptorImages.h>

#define enableModifiedQUICCI false

float distance(float3 &a, float3 &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

int main(int argc, const char** argv) {
    const unsigned int numberOfSearchResultsToGenerate = 1;


    arrrgh::parser parser("remeshedCorrespondenceFinder", "Find nearest neighbour images from vertices in a mesh to its remeshed counterpart.");
    const auto &originalMeshFile = parser.add<std::string>(
            "original-mesh", "The unmodified mesh.", '\0', arrrgh::Required, "");
    const auto &remeshedMeshFile = parser.add<std::string>(
            "remeshed-mesh", "The same mesh as the original, but remeshed.", '\0', arrrgh::Required, "");
    const auto &forceGPU = parser.add<int>(
            "force-gpu", "Index of the GPU device to use for kernels.", '\0', arrrgh::Optional, -1);
    const auto &outputFile = parser.add<std::string>(
            "output-file", "Path to a PNG file to which to write the nearest neighbour comparisons.", '\0', arrrgh::Optional, "NONE_SELECTED");
    const auto& imageStartIndex = parser.add<int>(
            "image-start-index", "The index of the first image from which to generate an image.", '\0', arrrgh::Optional, 0);
    const auto &imageLimit = parser.add<int>(
            "image-limit", "Limit the number of images to dump.", '\0', arrrgh::Optional, 5000);
    const auto &supportRadius = parser.add<float>(
            "support-radius", "Support radius of images to generate.", '\0', arrrgh::Optional, 1);
    const auto &showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);

    try {
        parser.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if (showHelp.value()) {
        return 0;
    }

    cudaDeviceProp gpuProperties = ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());

    std::cout << "Build info: " << GitMetadata::CommitSHA1() << ", by " << GitMetadata::AuthorName() << " on " << GitMetadata::CommitDate() << std::endl;

    ShapeDescriptor::cpu::Mesh originalMesh = ShapeDescriptor::utilities::loadMesh(originalMeshFile.value(), true);
    ShapeDescriptor::gpu::Mesh originalMeshGPU = ShapeDescriptor::copy::hostMeshToDevice(originalMesh);
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> originalDescriptorOriginsGPU = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(originalMeshGPU);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> originalDescriptorOrigins = ShapeDescriptor::copy::deviceArrayToHost(originalDescriptorOriginsGPU);
#if enableModifiedQUICCI
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> riciDescriptors = ShapeDescriptor::gpu::generateRadialIntersectionCountImages(originalMeshGPU, originalDescriptorOriginsGPU, supportRadius.value());
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> originalDescriptorsGPU = convertRICIToModifiedQUICCI(riciDescriptors);
    ShapeDescriptor::free::array(riciDescriptors);
#else
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> originalDescriptorsGPU = ShapeDescriptor::gpu::generateQUICCImages(originalMeshGPU, originalDescriptorOriginsGPU, supportRadius.value());
#endif
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> originalDescriptors = ShapeDescriptor::copy::deviceArrayToHost(originalDescriptorsGPU);
    ShapeDescriptor::free::array(originalDescriptorOriginsGPU);
    ShapeDescriptor::free::array(originalDescriptorsGPU);
    ShapeDescriptor::free::mesh(originalMeshGPU);

    std::cout << "Original done." << std::endl;
    ShapeDescriptor::cpu::Mesh remeshedMesh = ShapeDescriptor::utilities::loadMesh(remeshedMeshFile.value(), true);
    ShapeDescriptor::gpu::Mesh remeshedMeshGPU = ShapeDescriptor::copy::hostMeshToDevice(remeshedMesh);
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> remeshedDescriptorOriginsGPU = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(remeshedMeshGPU);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> remeshedDescriptorOrigins = ShapeDescriptor::copy::deviceArrayToHost(remeshedDescriptorOriginsGPU);
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> remeshedDescriptorsGPU = ShapeDescriptor::gpu::generateQUICCImages(remeshedMeshGPU, remeshedDescriptorOriginsGPU, supportRadius.value());
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> remeshedDescriptors = ShapeDescriptor::copy::deviceArrayToHost(remeshedDescriptorsGPU);
    ShapeDescriptor::free::array(remeshedDescriptorOriginsGPU);
    ShapeDescriptor::free::array(remeshedDescriptorsGPU);
    ShapeDescriptor::free::mesh(remeshedMeshGPU);
    std::cout << "Remeshed done." << std::endl;

    unsigned int imagesToGenerate = originalDescriptorOrigins.length;

    imagesToGenerate = std::max<int>(0, std::min<int>(int(imagesToGenerate) - imageStartIndex.value(), imageLimit.value()));

    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> originalReorderedDescriptors(imagesToGenerate);
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> remeshedReorderedDescriptors(imagesToGenerate);
#pragma omp parallel for
    for(unsigned int image = 0; image < imagesToGenerate; image++) {
        ShapeDescriptor::OrientedPoint origin = originalDescriptorOrigins.content[imageStartIndex.value() + image];

        // Initialise to any vertex
        unsigned int closestVertexIndex = 0;
        float closestVertexDistance = distance(origin.vertex, remeshedDescriptorOrigins.content[closestVertexIndex].vertex);

        for(unsigned int remeshedImage = 0; remeshedImage < remeshedDescriptorOrigins.length; remeshedImage++) {
            float distanceToVertex = distance(origin.vertex, remeshedDescriptorOrigins.content[remeshedImage].vertex);
            if(distanceToVertex < closestVertexDistance) {
                closestVertexIndex = remeshedImage;
                closestVertexDistance = distanceToVertex;
            }
        }

        originalReorderedDescriptors.content[image] = originalDescriptors.content[imageStartIndex.value() + image];
        remeshedReorderedDescriptors.content[image] = remeshedDescriptors.content[closestVertexIndex];
    }

    ShapeDescriptor::dump::descriptorComparisonImage(outputFile.value(), originalReorderedDescriptors, remeshedReorderedDescriptors, {0, nullptr}, 50);

}