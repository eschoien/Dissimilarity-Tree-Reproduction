#include <arrrgh.hpp>
#include <experimental/filesystem>
#include <projectSymmetry/clustering/ClusteredQuiccImage.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <fstream>
#include <set>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/types/ImageSearchResults.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageSearcher.cuh>
#include <projectSymmetry/visualisation/SymmetryVisualiser.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("alignmentexperiment", "Experimentation program to determine a good method of surface alignment.");
    const auto& queryFile = parser.add<std::string>(
            "query-file", "A portion of the haystack file.", '\0', arrrgh::Required, "");
    const auto& haystackFile = parser.add<std::string>(
            "haystack-file", "The object in which the query file should be located", '\0', arrrgh::Required, "");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    std::cout << "Loading query file " << queryFile.value() << "..." << std::endl;
    ShapeDescriptor::cpu::Mesh queryMesh = ShapeDescriptor::utilities::loadMesh(queryFile.value());
    std::cout << "Loading haystack file " << haystackFile.value() << "..." << std::endl;
    ShapeDescriptor::cpu::Mesh haystackMesh = ShapeDescriptor::utilities::loadMesh(haystackFile.value(), true);
    std::cout << haystackMesh.vertexCount << std::endl;

    std::cout << "Copying meshes to GPU.." << std::endl;
    ShapeDescriptor::gpu::Mesh gpuQueryMesh = ShapeDescriptor::copy::hostMeshToDevice(queryMesh);
    ShapeDescriptor::gpu::Mesh gpuHaystackMesh = ShapeDescriptor::copy::hostMeshToDevice(haystackMesh);

    std::cout << "Computing descriptor origins.." << std::endl;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> queryOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(gpuQueryMesh);
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> haystackOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(gpuHaystackMesh);

    const float supportRadius = 150;

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::cout << "Computing descriptors.." << std::endl;
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> queryDescriptors =
            ShapeDescriptor::gpu::generateQUICCImages(gpuQueryMesh, queryOrigins, supportRadius);
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors =
            ShapeDescriptor::gpu::generateQUICCImages(gpuHaystackMesh, haystackOrigins, supportRadius);

    std::cout << "Computing search results.." << std::endl;
    ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<unsigned int>> searchResults = ShapeDescriptor::gpu::findQUICCImagesInHaystack(queryDescriptors, haystackDescriptors);

    std::cout << "Visualising results.." << std::endl;
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> cpuQueryDescriptors = ShapeDescriptor::copy::deviceArrayToHost(queryDescriptors);

    ShapeDescriptor::cpu::Mesh combinedMesh(queryMesh.vertexCount + haystackMesh.vertexCount);
    std::copy(haystackMesh.vertices, haystackMesh.vertices + haystackMesh.vertexCount, combinedMesh.vertices);
    std::copy(queryMesh.vertices, queryMesh.vertices + queryMesh.vertexCount, combinedMesh.vertices + haystackMesh.vertexCount);
    std::copy(haystackMesh.normals, haystackMesh.normals + haystackMesh.vertexCount, combinedMesh.normals);
    std::copy(queryMesh.normals, queryMesh.normals + queryMesh.vertexCount, combinedMesh.normals + haystackMesh.vertexCount);

    for(unsigned int i = 0; i < queryMesh.vertexCount; i++) {
        combinedMesh.vertices[haystackMesh.vertexCount + i] = combinedMesh.vertices[haystackMesh.vertexCount + i] + ShapeDescriptor::cpu::float3(200, 0, 0);
    }

    visualise(combinedMesh, searchResults, 0.01, cpuQueryDescriptors);

    

    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    std::cout << std::endl << "Done." << std::endl;
}