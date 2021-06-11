#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/spinImageGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/dump/descriptorImages.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/kernels/meshSampler.cuh>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <projectSymmetry/descriptors/binaryRICIConverter.h>
#include <arrrgh.hpp>

int main(int argc, const char** argv) {
    arrrgh::parser parser("imagerenderer", "Generate RICI or spin images from an input object and dump them into a PNG file");
    const auto& inputFile = parser.add<std::string>(
            "input", "The location of the input OBJ model file.", '\0', arrrgh::Required, "");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);
    const auto& forceGPU = parser.add<int>(
            "force-gpu", "Force using the GPU with the given ID", 'b', arrrgh::Optional, -1);
    const auto& spinImageWidth = parser.add<float>(
            "support-radius", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, 1.0f);
    const auto& imageStartIndex = parser.add<int>(
            "image-start-index", "The index of the first image from which to generate an image.", '\0', arrrgh::Optional, 0);
    const auto& imageLimit = parser.add<int>(
            "image-limit", "The maximum number of images to generate (in order to limit image size)", '\0', arrrgh::Optional, 5000);
    const auto& imagesPerRow = parser.add<int>(
            "images-per-row", "The number of images the output image should contain per row", '\0', arrrgh::Optional, 50);
    const auto& outputFile = parser.add<std::string>(
            "output", "The location of the PNG file to write to", '\0', arrrgh::Optional, "out.png");

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

    if(forceGPU.value() != -1) {
        std::cout << "Forcing GPU " << forceGPU.value() << std::endl;
        ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());
    }

    std::cout << "Loading mesh file.." << std::endl;
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(inputFile.value(), true);
    ShapeDescriptor::gpu::Mesh deviceMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);
    std::cout << "    Object has " << mesh.vertexCount << " vertices" << std::endl;

    std::cout << "Locating unique vertices.." << std::endl;

    size_t backupSize = deviceMesh.vertexCount;
    if(imageLimit.value() != -1) {
        deviceMesh.vertexCount = 10*imageLimit.value();
    }

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> spinOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(deviceMesh);

    if(imageLimit.value() != -1) {
        deviceMesh.vertexCount = backupSize;
        spinOrigins.length = std::min<int>(spinOrigins.length, imageLimit.value());
    }

    std::cout << "Generating images.. (this can take a while)" << std::endl;
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors =
            ShapeDescriptor::gpu::generateRadialIntersectionCountImages(
            deviceMesh,
            spinOrigins,
            spinImageWidth.value());

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> convertedDescriptors = convertRICIToModifiedQUICCI(descriptors);

    std::cout << "Dumping results.. " << std::endl;
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> hostDescriptors =
            ShapeDescriptor::copy::deviceArrayToHost(convertedDescriptors);
    hostDescriptors.length = std::max<int>(0, std::min<int>(int(hostDescriptors.length) - imageStartIndex.value(), imageLimit.value()));
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> offsetHostDescriptors = hostDescriptors;
    offsetHostDescriptors.content += imageStartIndex.value();
    ShapeDescriptor::dump::descriptors(offsetHostDescriptors, outputFile.value(), imagesPerRow.value());
    delete[] hostDescriptors.content;

    cudaFree(descriptors.content);

    ShapeDescriptor::free::mesh(mesh);
    ShapeDescriptor::gpu::freeMesh(deviceMesh);
}