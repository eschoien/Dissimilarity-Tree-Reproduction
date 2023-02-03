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
#include <shapeDescriptor/utilities/free/array.h>
#include <projectSymmetry/descriptors/quicciStats.h>
#include <projectSymmetry/descriptors/quicciStatsCPU.h>


void testDescriptorSignature(const std::experimental::filesystem::path &sourceDirectory, const int fileID, const int descriptorID, const int numberOfPermutations) {

    // vector for signatures
    std::vector<int> signatures;
    std::vector<int>* signaturesPtr = &signatures;

    std::cout << "Reading files from: " << sourceDirectory << std::endl;
    std::cout << "Permutations#: " << numberOfPermutations << std::endl;

    std::vector<std::vector<int>> permutations = create_permutations(numberOfPermutations);

    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(sourceDirectory);

    // Get the specified descriptor
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::read::QUICCIDescriptors(haystackFiles.at(fileID));
    ShapeDescriptor::QUICCIDescriptor testDescriptor = descriptors.content[descriptorID];

    // Compute and place signatures into vector

    computeDescriptorSignature(testDescriptor, signaturesPtr, permutations);

    std::cout << "Object# " << fileID << " Descriptor# " << descriptorID << std::endl;
    ShapeDescriptor::print::quicciDescriptor(testDescriptor);

    std::cout << "Signature: " << std::endl;
    for (int s = 0; s < signatures.size(); s++) {
        std::cout << signatures[s] << ' ';
    }
    std::cout << std::endl;

    std::cout << "Permutations: " << std::endl;
    for (int p = 0; p < permutations.size(); p++) {
        std::cout << p << ": ";
        for (int i = 0; i < 100; i++) {
            std::cout << permutations[p][i] << " ";
        }
        std::cout << std::endl;
    }
}
    

int main(int argc, const char **argv) {
    arrrgh::parser parser("descriptorSignatureTest", "Create indexes for QUICCI images.");
    const auto& sourceDirectory = parser.add<std::string>(
        "quicci-dump-directory", "The directory where binary dump files of QUICCI images are stored that should be indexed.", '\0', arrrgh::Required, "output/descriptors/complete_objects_32x32");
    const auto& fileID = parser.add<int>(
        "file-id", "Object file id", '\0', arrrgh::Optional, 0);
    const auto& descriptorID = parser.add<int>(
        "descriptor-id", "Descriptor id", '\0', arrrgh::Optional, 0);
    const auto& numberOfPermutations = parser.add<int>(
        "permutation-count", "Number of Minhash functions (signature length)", '\0', arrrgh::Optional, 10);

    const auto &showHelp = parser.add<bool>(
        "help", "Show this help message.", 'h', arrrgh::Optional, false);

    // Show help if desired
    if (showHelp.value()) {
        return 0;
    }
    
    testDescriptorSignature(sourceDirectory.value(), fileID.value(), descriptorID.value(), numberOfPermutations.value());

    return 0;
}