#include <arrrgh.hpp>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/dump/QUICCIDescriptors.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <projectSymmetry/descriptors/binaryRICIConverter.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("descriptorDumper", "Do symmetry detection or something along those lines.");
    const auto& showHelp = parser.add<bool>("help", "Show this help message.", 'h', arrrgh::Optional, false);
    const auto& meshFile = parser.add<std::string>("input-file", "Location of the file which should be queried", '\0', arrrgh::Optional, "NOT_SELECTED");
    const auto &forceGPU = parser.add<int>("force-gpu", "Index of the GPU device to use for search kernels.", '\0', arrrgh::Optional, -1);
    const auto& outputFile = parser.add<std::string>("output-file", "Location where to dump the produced QUICCI descriptors", '\0', arrrgh::Optional, "NOT_SELECTED");
    const auto& supportRadius = parser.add<float>("support-radius", "The support radius to use during rendering", '\0', arrrgh::Optional, 1);
    const auto& listGPUs = parser.add<bool>("list-gpus", "Print a list of available GPUs to stdout.", '\0', arrrgh::Optional, false);

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

    if(listGPUs.value()) {
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));

        std::cout << "Found " << deviceCount << " devices:" << std::endl;

        for(int i = 0; i < deviceCount; i++)
        {
            cudaDeviceProp deviceProperties;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProperties, i));

            std::cout << "\t- " << deviceProperties.name << " (ID " << i << ")" << std::endl;
        }
        return 0;
    }

    cudaDeviceProp gpuProperties = ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());

    ShapeDescriptor::cpu::Mesh inputMesh = ShapeDescriptor::utilities::loadMesh(meshFile.value(), true);

    ShapeDescriptor::gpu::Mesh device_inputMesh = ShapeDescriptor::copy::hostMeshToDevice(inputMesh);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> spinOrigins =
            ShapeDescriptor::utilities::generateSpinOriginBuffer(device_inputMesh);

    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors =
            ShapeDescriptor::gpu::generateRadialIntersectionCountImages(device_inputMesh, spinOrigins, supportRadius.value());

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> convertedDescriptors =
            convertRICIToModifiedQUICCI(descriptors);

    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> hostDescriptors =
            ShapeDescriptor::copy::deviceArrayToHost(convertedDescriptors);

    ShapeDescriptor::dump::raw::QUICCIDescriptors(outputFile.value(), hostDescriptors, 0);

    ShapeDescriptor::free::mesh(inputMesh);
    ShapeDescriptor::free::mesh(device_inputMesh);
    ShapeDescriptor::free::array(spinOrigins);
    ShapeDescriptor::free::array(convertedDescriptors);
    ShapeDescriptor::free::array(descriptors);
}