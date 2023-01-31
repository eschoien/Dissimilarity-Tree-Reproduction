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
#include "Signature.h"
#include "SignatureIO.h"
#include "SignatureBuilder.h"

std::vector<std::vector<int>> create_permutations(int numberOfPermutations) {

    std::vector<std::vector<int>> permutations;

    for (int n = 0; n < numberOfPermutations; n++) {

        std::vector<int> numbers;

        for (int i=0; i <= 1023; i++) {
            numbers.push_back(i);
        }

        std::random_shuffle(&numbers[0], &numbers[1024]);

        permutations.push_back(numbers);
    }

    return permutations;
}

std::vector<DescriptorSignature> buildSignaturesFromDumpDirectory(const std::experimental::filesystem::path &imageDumpDirectory, const std::experimental::filesystem::path &outputDirectory, const unsigned int number_of_permutations) {
    
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
        ObjectSignature* oSig = new ObjectSignature;
        oSig->file_id = i;
        // loop through descriptors for current object
        for(unsigned int j = 0; j < descriptors.length; j++) {
            DescriptorSignature dSig;
            dSig.descriptor_id = j;

            // ShapeDescriptor::print::quicciDescriptor(descriptors.content[j]);

            // perform minhash

            // loop through the different permutations
            for (unsigned int p = 0; p < number_of_permutations; p++) {

                std::vector<int> permutation = permutations[p];

                unsigned int m = 0;

                while (m < 1024) {
                    // std::cout << j << " " << m << " " << (permutation[m] / 32) << std::endl;
                    if (descriptors.content[j].contents[(permutation[m] / 32)] & (1 << (permutation[m] % 32))) {
                        break;
                    }   
                    m++;

                }
                //m is now the signature of this descriptor for the current permutation
                dSig.signatures.push_back(m);
            }
            oSig->descriptorSignatures.push_back(dSig);
            // delete dSig;
        }
        writeSignatures(*oSig, outputDirectory);
        std::cout << oSig->file_id << std::endl;
        delete oSig;
        ShapeDescriptor::free::array(descriptors);
    }

    // Measure execution time
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl << "MinHash signature construction complete. " << std::endl;
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    return descriptorSignatures;
}