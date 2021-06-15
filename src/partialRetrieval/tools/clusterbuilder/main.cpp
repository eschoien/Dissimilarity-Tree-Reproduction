#include <arrrgh.hpp>
#include <experimental/filesystem>
#include <fstream>
#include <set>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <projectSymmetry/clustering/ClusteredQuiccImage.h>
#include <projectSymmetry/clustering/ClusterBuilder.h>
#include <projectSymmetry/clustering/ClusterIO.h>

// Capacity: SHREC2016/all: 36491670
// Capacity: SHREC2016/100: 9993441

int main(int argc, const char** argv) {
    arrrgh::parser parser("clusterbuilder", "Create indexes for QUICCI images.");
    const auto& indexDirectory = parser.add<std::string>(
            "index-directory", "The directory where the index should be stored.", '\0', arrrgh::Optional, "");
    const auto& sourceDirectory = parser.add<std::string>(
            "quicci-dump-directory", "The directory where binary dump files of QUICCI images are stored that should be indexed.", '\0', arrrgh::Optional, "");
    const auto &forceGPU = parser.add<int>(
            "force-gpu", "Index of the GPU device to use for search kernels.", '\0', arrrgh::Optional, -1);
    const auto& forceCPU = parser.add<bool>(
            "force-cpu", "Index is built using the CPU rather than the GPU. Much slower, but not all datasets fit in VRAM.", '\0', arrrgh::Optional, false);
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

    cudaDeviceProp gpuProperties;
    if(!forceCPU.value()) {
        gpuProperties = ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());
    }

    std::cout << "Building index from files in " << sourceDirectory.value() << "..." << std::endl;

    Cluster* cluster = buildClusterFromDumpDirectory(sourceDirectory.value(), indexDirectory.value(), 32, forceCPU.value());

    std::cout << "Writing cluster file.." << std::endl;

    writeCluster(cluster, cluster::path(indexDirectory.value()) / "index.dat");

    std::cout << std::endl << "Done." << std::endl;
}