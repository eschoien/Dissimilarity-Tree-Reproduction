#include <arrrgh.hpp>
#include <iostream>
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
#include "Permutation.h"

SignatureIndex buildSignaturesFromDumpDirectory(const std::experimental::filesystem::path &imageDumpDirectory, const std::experimental::filesystem::path &outputDirectory, const unsigned int number_of_permutations) {
    
    SignatureIndex signatureIndex;
    signatureIndex.objectCount = 0;
    signatureIndex.numPermutations = number_of_permutations;


    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(imageDumpDirectory);

    /* TODO: Look into atomic index, should we use this?
    std::atomic<unsigned int> nextStartIndex;
    nextStartIndex = 0;
    */

    // generate minhash permutations
    signatureIndex.permutations = create_permutations(number_of_permutations);

    std::cout << "Processing descriptors.." << std::endl;

    // loop through all objects
    for(unsigned int i = 0; i < haystackFiles.size(); i++) {

        // Start time for current object
        std::chrono::steady_clock::time_point objectStartTime = std::chrono::steady_clock::now();

        // loads all the descriptors for current object
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::read::QUICCIDescriptors(haystackFiles.at(i));

        ObjectSignature* objectSignature = new ObjectSignature;

        objectSignature->file_id = i + 1;
        signatureIndex.objectCount++;
        
        // loop through descriptors for current object
        for(unsigned int j = 0; j < descriptors.length; j++) {
            DescriptorSignature descriptorSignature;
            descriptorSignature.descriptor_id = j + 1;
            computeDescriptorSignature(descriptors.content[j], &(descriptorSignature.signatures), signatureIndex.permutations);
            objectSignature->descriptorSignatures.push_back(descriptorSignature);
            //delete descriptorSignature;
        }

        //Write object signature to file
        writeSignatures(*objectSignature, outputDirectory, number_of_permutations);

        // End time for current object
        std::chrono::steady_clock::time_point objectEndTime = std::chrono::steady_clock::now();
        auto objectDuration = std::chrono::duration_cast<std::chrono::milliseconds>(objectEndTime - objectStartTime);

        // ----- OUTPUT -----
        std::cout << "ObjectFileId: " << objectSignature->file_id << std::endl;
        std::cout << descriptors.length << " descriptors" << std::endl;
        std::cout << objectSignature->descriptorSignatures.size() << " signatures" << std::endl;        
        std::cout << float(objectDuration.count()) / 1000.0f << " seconds" << std::endl;
        std::cout << std::endl;
        // ------------------

        delete objectSignature;
        ShapeDescriptor::free::array(descriptors);
    }

    // Measure total execution time
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl << "MinHash signature construction complete. " << std::endl;
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    return signatureIndex;
}