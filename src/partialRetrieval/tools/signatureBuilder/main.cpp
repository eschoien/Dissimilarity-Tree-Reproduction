#include <arrrgh.hpp>
#include <iostream>
#include <atomic>
#include <string>
#include <shapeDescriptor/utilities/fileutils.h>
#include <projectSymmetry/lsh/Signature.h>
#include <projectSymmetry/lsh/SignatureIO.h>
#include <projectSymmetry/lsh/SignatureBuilder.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("signatureBuilder", "Create indexes for QUICCI images.");
    const auto& indexDirectory = parser.add<std::string>(
        "index-directory", "The directory where the signature file should be stored.", '\0', arrrgh::Optional, "");
    const auto& sourceDirectory = parser.add<std::string>(
        "quicci-dump-directory", "The directory where binary dump files of QUICCI images are stored that should be indexed.", '\0', arrrgh::Optional, "");
    const auto& numberOfPermutations = parser.add<int>(
        "permutation-count", "The number of MInhash permutations / signature length", '\0', arrrgh::Optional, 10);
    const auto &descriptorsPerObjectLimit = parser.add<int>(
        "descriptorsPerObjectLimit", "descriptorsPerObjectLimit", '\0', arrrgh::Optional, 2000);
    const auto &seed = parser.add<int>(
        "randomSeed", "Random seed to use for determining the order of query images to visit.", '\0', arrrgh::Optional, 725948161);
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

    std::cout << "Number of Minhash functions / permutations: " << numberOfPermutations.value() << std::endl;

    SignatureIndex signatureIndex = buildSignaturesFromDumpDirectory(sourceDirectory.value(), std::experimental::filesystem::path(indexDirectory.value()) / "minhash_signatures/", numberOfPermutations.value(), descriptorsPerObjectLimit.value(), seed.value());

    std::cout << "Writing Signature index file.." << std::endl;

    writeSignatureIndex(signatureIndex, std::experimental::filesystem::path(indexDirectory.value()) / "index.dat");

    std::cout << std::endl << "Done." << std::endl;
}