#include <arrrgh.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <atomic>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/read/QUICCIDescriptors.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <projectSymmetry/descriptors/quicciStats.h>
#include <projectSymmetry/descriptors/quicciStatsCPU.h>
#include "DescriptorSignature.h"


// Function for creating n number of permuations of integers 1-1024.

std::vector<std::vector<int>> create_permutations(int numberOfPermutations) {

    std::vector<std::vector<int>> permutations;

    for (int n = 0; n < numberOfPermutations; n++) {

        std::vector<int> numbers;

        for (int i=1; i <= 1024; i++) {
            numbers.push_back(i);
        }

        std::random_shuffle(&numbers[0], &numbers[1024]);

        permutations.push_back(numbers);
    }

    return permutations;
}

std::vector<DescriptorSignature> buildSignaturesFromDumpDirectory(const std::experimental::filesystem::path &imageDumpDirectory, const unsigned int number_of_permutations) {
    
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(imageDumpDirectory);

    /*
    std::cout << "Counting images.." << std::endl;
    size_t imageCountToIndex = 0;
    for(unsigned int i = 0; i < haystackFiles.size(); i++) {
        ShapeDescriptor::QUICCIDescriptorFileHeader header = ShapeDescriptor::read::QuicciDescriptorFileHeader(haystackFiles.at(i));
        imageCountToIndex += header.imageCount;
    }
    std::cout << "Found " << imageCountToIndex << " images in directory" << std::endl;

    std::atomic<unsigned int> nextStartIndex;
    nextStartIndex = 0;
    */

    // generate minhash permutations
    std::vector<std::vector<int>> permutations = create_permutations(number_of_permutations);

    // vector for all signatures
    std::vector<DescriptorSignature> descriptorSignatures;

    std::cout << "Loading descriptors.." << std::endl;

    // loop through all objects
    for(unsigned int i = 0; i < haystackFiles.size(); i++) {

        // array of descriptors for object
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::read::QUICCIDescriptors(haystackFiles.at(i));
        
        // loop through descriptors for current object
        for(unsigned int j = 0; j < descriptors.length; j++) {
            DescriptorSignature dSig;
            dSig.file_id = i;
            dSig.descriptor_id = j;

            // ShapeDescriptor::print::quicciDescriptor(descriptors.content[j]);

            // perform minhash

            // loop through the different permutations
            for (unsigned int p = 0; p < number_of_permutations; p++) {

                std::vector<int> permutation = permutations[p];

                unsigned int m = 0;

                while (true) {
                    if (descriptors.content[j].contents[(permutation[m] / 32)] & (1 << (permutation[m] % 32))) {
                        break;
                    }   
                    m++;

                }
                //m is now the signature of th50is descriptor for the current permutation
                dSig.signatures.push_back(m);
            }

            descriptorSignatures.push_back(dSig);
        }

        ShapeDescriptor::free::array(descriptors);
    }

    // Measure execution time
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl << "MinHash signature construction complete. " << std::endl;
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    return descriptorSignatures;
}

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

    // Cluster* cluster = buildClusterFromDumpDirectory(sourceDirectory.value(), indexDirectory.value(), 32, forceCPU.value());
    // TODO: Generate the signatures ...

    unsigned int number_of_permutations = 10;

    std::vector<DescriptorSignature> descriptorSignatures = buildSignaturesFromDumpDirectory(sourceDirectory.value(), number_of_permutations);

    // print signatures

    // works but gives segfault for too many descriptors?
    for (int i = 0; i < descriptorSignatures.size(); i++) {

        std::cout << descriptorSignatures[i].file_id << "-";
        std::cout << descriptorSignatures[i].descriptor_id << ": ";

        for (int j = 0; j < number_of_permutations; j++) {
            std::cout << descriptorSignatures[i].signatures[j] << " ";
        }
        std::cout << " " << std::endl;
    }

    std::cout << "Writing cluster file.." << std::endl;

    // writeCluster(cluster, cluster::path(indexDirectory.value()) / "index.dat");
    // TODO: Write signature file ... (not implemented)

    std::cout << std::endl << "Done." << std::endl;
}