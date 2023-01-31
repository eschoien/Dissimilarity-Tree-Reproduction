#include <arrrgh.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <atomic>
#include <string> 
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/read/QUICCIDescriptors.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <projectSymmetry/descriptors/quicciStats.h>
#include <projectSymmetry/descriptors/quicciStatsCPU.h>
#include <projectSymmetry/lsh/Signature.h>
#include <projectSymmetry/lsh/SignatureIO.h>
#include <projectSymmetry/lsh/SignatureBuilder.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("clusterbuilder", "Create indexes for QUICCI images.");
    const auto& indexDirectory = parser.add<std::string>(
            "index-directory", "The directory where the signature file should be stored.", '\0', arrrgh::Optional, "");
    const auto& sourceDirectory = parser.add<std::string>(
            "quicci-dump-directory", "The directory where binary dump files of QUICCI images are stored that should be indexed.", '\0', arrrgh::Optional, "");
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

    std::cout << "Computing signatures from files in " << sourceDirectory.value() << "..." << std::endl;


    unsigned int number_of_permutations = 10;

    std::vector<DescriptorSignature> descriptorSignatures = buildSignaturesFromDumpDirectory(sourceDirectory.value(), indexDirectory.value(), number_of_permutations);

    // print signatures

    // works but gives segfault for too many descriptors?
    // for (int i = 0; i < descriptorSignatures.size(); i++) {

    //     std::cout << descriptorSignatures[i].file_id << "-";
    //     std::cout << descriptorSignatures[i].descriptor_id << ": ";

    //     for (int j = 0; j < number_of_permutations; j++) {
    //         std::cout << descriptorSignatures[i].signatures[j] << " ";
    //     }
    //     std::cout << " " << std::endl;
    // }

    std::cout << "Writing cluster file.." << std::endl;

    // writeCluster(cluster, cluster::path(indexDirectory.value()) / "index.dat");
    // TODO: Write signature file ... (not implemented)
    // written to file during buildSignatures

    std::cout << std::endl << "Done." << std::endl;
}