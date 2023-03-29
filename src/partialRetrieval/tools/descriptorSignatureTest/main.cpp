#include <arrrgh.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <shapeDescriptor/utilities/read/QUICCIDescriptors.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <projectSymmetry/lsh/Permutation.h>
#include <projectSymmetry/lsh/Signature.h>
#include <atomic>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <projectSymmetry/descriptors/quicciStats.h>
#include <projectSymmetry/descriptors/quicciStatsCPU.h>

int main(int argc, const char **argv) {
    arrrgh::parser parser("descriptorSignatureTest", "Print Minhash signature and permutations for QUICCI descriptor.");
    const auto& sourceDirectory = parser.add<std::string>(
        "quicci-dump-directory", "The directory of binary dump files of QUICCI images that should be used.", '\0', arrrgh::Required, "output/descriptors/complete_objects_32x32");
    const auto& fileID = parser.add<int>(
        "file-id", "Object file id", '\0', arrrgh::Optional, 0);
    const auto& descriptorID = parser.add<int>(
        "descriptor-id", "Descriptor id", '\0', arrrgh::Optional, 0);
    const auto& numberOfPermutations = parser.add<int>(
        "permutation-count", "Number of Minhash functions (signature length)", '\0', arrrgh::Optional, 10);
    const auto &seed = parser.add<int>(
        "randomSeed", "Random seed to use for determining the order of query images to visit.", '\0', arrrgh::Optional, 725948161);
    const auto& showHelp = parser.add<bool>(
        "help", "Show this help message.", 'h', arrrgh::Optional, false);

    try {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if (showHelp.value()) {
        return 0;
    }

    std::cout << "Reading files from directory: " << sourceDirectory.value() << std::endl;

    // Get the specified descriptor
    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(sourceDirectory.value());
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::read::QUICCIDescriptors(haystackFiles.at(fileID.value()));
    ShapeDescriptor::QUICCIDescriptor testDescriptor = descriptors.content[descriptorID.value()];

    //std::vector<std::vector<int>> permutations = create_permutations(numberOfPermutations.value(), seed.value());



    std::vector<int> signature;
    computeDescriptorSignature(testDescriptor, &signature, numberOfPermutations);

    // ----- OUTPUT -----
    std::cout << "Object# " << fileID.value() << " Descriptor# " << descriptorID.value() << std::endl;
    ShapeDescriptor::print::quicciDescriptor(testDescriptor);

    std::cout << "Signature: " << std::endl;
    lsh::print::signature(signature);

    // std::cout << "Permutations: " << std::endl;
    // lsh::print::permutations(permutation, 50);
    // ------------------

    return 0;
}