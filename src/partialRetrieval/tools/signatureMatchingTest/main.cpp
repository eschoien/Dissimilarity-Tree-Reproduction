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
#include <projectSymmetry/lsh/SignatureIO.h>

int main(int argc, const char **argv) {
    arrrgh::parser parser("signatureMatchingTest", "Match descriptor signatures.");
    const auto& descriptorDirectory = parser.add<std::string>(
        "quicci-dump-directory", "The directory where binary dump files of QUICCI images are stored that should be used as sample.", '\0', arrrgh::Required, "");
    const auto& signatureDirectory = parser.add<std::string>(
        "signature-dump-directory", "The directory where binary dump files of objects with QUICCI signatures are stored that should be searched.", '\0', arrrgh::Required, "");
    const auto& fileID = parser.add<int>(
        "file-id", "Object file id", '\0', arrrgh::Optional, 0);
    const auto& descriptorID = parser.add<int>(
        "descriptor-id", "Descriptor id", '\0', arrrgh::Optional, 0);

    const auto &showHelp = parser.add<bool>(
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
    if (showHelp.value()) {
        return 0;
    }

    // load query descriptor
    std::cout << "Reading files from directory: " << descriptorDirectory.value() << std::endl;
    std::vector<std::experimental::filesystem::path> descriptorFiles = ShapeDescriptor::utilities::listDirectory(descriptorDirectory.value());
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::read::QUICCIDescriptors(descriptorFiles.at(fileID.value()-1));
    ShapeDescriptor::QUICCIDescriptor queryDescriptor = descriptors.content[descriptorID.value()];

    ShapeDescriptor::print::quicciDescriptor(queryDescriptor);

    // test for more than one descriptor ... loop

    // load permutations
    SignatureIndex *signatureIndex = readSignatureIndex("output/lsh/index.dat");
    std::vector<std::vector<int>> permutations = signatureIndex->permutations;

    // compute signatures
    std::vector<int> querySignature;
    computeDescriptorSignature(queryDescriptor, &querySignature, permutations);

    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(signatureDirectory.value());
    
    double JACCARD_THRESHOLD = 0.5; // * signatureIndex->numPermutations;
    unsigned int descriptorsPerObjectLimit = 2000;

    std::vector<int> scores = {0,0,0,0,0,0,0,0,0,0,0};
    std::vector<int> objectScores(haystackFiles.size(), 0);

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    
    // loop through object signatures
    for(unsigned int i = 0; i < haystackFiles.size(); i++) {
    
        ObjectSignature* objectSignature = new ObjectSignature;
        objectSignature = readSignature(haystackFiles.at(i), signatureIndex->numPermutations);

        // loop descriptor signatures
        // for (unsigned int j = 0; j < (objectSignature->descriptorSignatures).size(); j++) {
        for (unsigned int j = 0; j < descriptorsPerObjectLimit; j++) {

            double jaccardSimilarity = computeJaccardSimilarity(querySignature, objectSignature->descriptorSignatures[j].signatures);

            if (jaccardSimilarity >= JACCARD_THRESHOLD) {
                objectScores[objectSignature->file_id-1]++;        
            }

            scores[jaccardSimilarity*querySignature.size()]++;
        }
        delete objectSignature;
    }


    // TIME
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;
    
    std::cout << "Signature matching scores: " << std::endl;
    lsh::print::signature(scores);
    
    std::cout << "Object scores: " << std::endl;
    lsh::print::signature(objectScores);

    for (int i = 0; i < objectScores.size(); i++) {
        std::cout << "File-" << i+1<< ": " << objectScores[i] << std::endl;
    }
    int sum = 0;
    std::for_each(objectScores.begin(), objectScores.end(), [&] (int n) {sum += n;});
    std::cout << "Object scores sum: " << sum << std::endl;
    std::cout << "Best matching object: " << std::distance(objectScores.begin(),std::max_element(objectScores.begin(), objectScores.end())) + 1 << std::endl;

    return 0;
}


/* OLD VERSION, DONT DELETE
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
#include <projectSymmetry/lsh/SignatureIO.h>

int main(int argc, const char **argv) {
    arrrgh::parser parser("signatureMatchingTest", "Match descriptor signatures.");
    const auto& descriptorDirectory = parser.add<std::string>(
        "quicci-dump-directory", "The directory where binary dump files of QUICCI images are stored that should be used as sample.", '\0', arrrgh::Required, "");
    const auto& signatureDirectory = parser.add<std::string>(
        "signature-dump-directory", "The directory where binary dump files of objects with QUICCI signatures are stored that should be searched.", '\0', arrrgh::Required, "");
    const auto& fileID = parser.add<int>(
        "file-id", "Object file id", '\0', arrrgh::Optional, 0);
    const auto& descriptorID = parser.add<int>(
        "descriptor-id", "Descriptor id", '\0', arrrgh::Optional, 0);

    const auto &showHelp = parser.add<bool>(
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
    if (showHelp.value()) {
        return 0;
    }

    // load query descriptor
    std::cout << "Reading files from directory: " << descriptorDirectory.value() << std::endl;
    std::vector<std::experimental::filesystem::path> descriptorFiles = ShapeDescriptor::utilities::listDirectory(descriptorDirectory.value());
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::read::QUICCIDescriptors(descriptorFiles.at(fileID.value()));
    ShapeDescriptor::QUICCIDescriptor queryDescriptor = descriptors.content[descriptorID.value()];

    // test for more than one descriptor ... loop

    // load permutations
    SignatureIndex *signatureIndex = readSignatureIndex("output/lsh/index.dat");
    std::vector<std::vector<int>> permutations = signatureIndex->permutations;

    // compute signatures
    std::vector<int> querySignature;
    computeDescriptorSignature(queryDescriptor, &querySignature, permutations);

    unsigned int bestMatchScore = 0;
    unsigned int bestMatchCount = 0;
    unsigned int bestMatchObject;
    unsigned int bestMatchDescriptor;
    std::vector<int> scores = {0,0,0,0,0,0,0,0,0,0,0};

    std::cout << "Query signature:" << std::endl;
    lsh::print::signature(querySignature);
    
    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(signatureDirectory.value());
    
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    // loop through object signatures
    for(unsigned int i = 0; i < haystackFiles.size(); i++) {
    
        ObjectSignature* objectSignature = new ObjectSignature;
        objectSignature = readSignature(haystackFiles.at(i), signatureIndex->permutations.size());

        // loop descriptor signatures
        // for (unsigned int j = 0; j < (objectSignature->descriptorSignatures).size(); j++) {
        for (unsigned int j = 0; j < 2000; j++) {

            std::vector<int> compareSignature = objectSignature->descriptorSignatures[j].signatures;

            //compute jaccard
            unsigned int matchScore = 0;

            for (unsigned int k = 0; k < querySignature.size(); k++) {
                if (querySignature[k] == compareSignature[k]) {
                    matchScore++;
                }
            }

            if (matchScore > bestMatchScore) {
                bestMatchScore = matchScore;
                bestMatchCount = 1;
                bestMatchObject = objectSignature->file_id;// + 1;
                bestMatchDescriptor = j;// + 1
                //bestMatchSignature. = compareSignature.copy()
            }
            else if (matchScore == bestMatchScore) {
                bestMatchCount++;
            }
            scores[matchScore]++;
        }
        delete objectSignature;
    }

    // TIME
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    // OUTPUT
    std::cout << "Query permutations length: " << querySignature.size() << std::endl;
    std::cout << "Dataset permutations length: " << (signatureIndex->permutations).size() << std::endl;
    std::cout << std::endl;

    std::cout << "Best match score: " << bestMatchScore << std::endl;
    std::cout << "Best match count: " << bestMatchCount << std::endl;
    std::cout << "First best match object: " << bestMatchObject - 1<< std::endl;
    std::cout << "First best match descriptor: " << bestMatchDescriptor << std::endl;
    lsh::print::signature(scores);
    std::cout << std::endl;

    return 0;
}
*/