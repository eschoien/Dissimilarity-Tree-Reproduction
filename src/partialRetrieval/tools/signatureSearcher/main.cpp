#include <arrrgh.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <atomic>
#include <string> 
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <projectSymmetry/descriptors/binaryRICIConverter.h>
#include <projectSymmetry/clustering/IndexQueryer.h>
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

void runSignatureQuery(
        std::experimental::filesystem::path queryFile,
        SignatureIndex *signatureIndex,
        float supportRadius,
        std::vector<std::experimental::filesystem::path> &haystackFiles
        ) {

  // --- load partial object and compute descriptors
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(queryFile, true);
    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(
            gpuMesh);

    // Compute the descriptor(s)
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> riciDescriptors =
            ShapeDescriptor::gpu::generateRadialIntersectionCountImages(gpuMesh, descriptorOrigins, supportRadius);

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = convertRICIToModifiedQUICCI(riciDescriptors);
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> queryDescriptors = ShapeDescriptor::copy::deviceArrayToHost(descriptors);
    // ----

    /*
    std::random_device rd("/dev/urandom");
    size_t randomSeed = seed != 0 ? seed : rd();
    std::minstd_rand0 generator{randomSeed};
    std::uniform_real_distribution<float> distribution(0, 1);
    */


    unsigned int descriptorsPerObjectLimit = 2000;
    double JACCARD_THRESHOLD = 0.5;

    

    std::vector<int> objectScores(haystackFiles.size(), 0);
   
    // Create partial object signatures
    ObjectSignature* queryObjectSignature = new ObjectSignature;
    
    std::string path_string = queryFile.string();
    std::size_t pos = path_string.find("/T");
    queryObjectSignature->file_id = std::stoi(path_string.substr(pos+2));

    std::cout << "Partial object: " << queryObjectSignature->file_id << std::endl;

    for(unsigned int i = 0; i < descriptorsPerObjectLimit; i++) {
            DescriptorSignature descriptorSignature;
            descriptorSignature.descriptor_id = i + 1;
            computeDescriptorSignature(queryDescriptors.content[i], &(descriptorSignature.signatures), signatureIndex->permutations);
            queryObjectSignature->descriptorSignatures.push_back(descriptorSignature);
    }


    // loop through signature index object signatures
    for(unsigned int i = 0; i < haystackFiles.size(); i++) {
        ObjectSignature* objectSignature = new ObjectSignature;
        objectSignature = readSignature(haystackFiles.at(i), signatureIndex->numPermutations);

        // loop through descripor signatures of signature index complete objects
        for (unsigned int j = 0; j < descriptorsPerObjectLimit; j++) {

            std::vector<int> candidateSignature =  objectSignature->descriptorSignatures[j].signatures;
            
            for(unsigned int k = 0; k < queryObjectSignature->descriptorSignatures.size(); k++) {

                std::vector<int> querySignature = queryObjectSignature->descriptorSignatures[k].signatures;

                double jaccardSimilarity = computeJaccardSimilarity(querySignature, candidateSignature);
            
                if (jaccardSimilarity >= JACCARD_THRESHOLD) {
                    objectScores[objectSignature->file_id-1]++;        
                }

            }
        delete objectSignature;
        }
    }   
    delete queryObjectSignature;

    // best matching object
    // returns the object with the highest number of descriptor signatures with jaccard similarity greater than threshold
    unsigned int bestMatch = std::distance(objectScores.begin(), std::max_element(objectScores.begin(), objectScores.end())) + 1;
    std::cout << "Best matching object" << bestMatch << std::endl;


}

// modified from another object search, needs further changes
int main(int argc, const char **argv) {
    arrrgh::parser parser("objectLshSearch", "Search for similar objects using LSH.");
    const auto &signatureDirectory = parser.add<std::string>(
        "signature-directory", "The directory containing the signatures to be queried.", '\0', arrrgh::Required, "");
    const auto &queryDirectory = parser.add<std::string>(
        "query-directory", "The directory containing meshes which should be used for querying the index.", '\0', arrrgh::Required, "");
    const auto &resultsPerQuery = parser.add<int>(
        "resultsPerQueryImage", "Number of search results to generate for each image in the query object.", '\0', arrrgh::Optional, 5);
    const auto &outputFile = parser.add<std::string>(
        "output-file", "Path to a JSON file to which to write the search results.", '\0', arrrgh::Optional, "NONE_SELECTED");
    const auto &seed = parser.add<int>(
        "randomSeed", "Random seed to use for determining the order of query images to visit.", '\0', arrrgh::Optional, 725948161);
    const auto &supportRadius = parser.add<float>(
        "support-radius", "Support radius to use for generating quicci descriptors.", '\0', arrrgh::Optional, 1.0);
    const auto &consensusThreshold = parser.add<int>(
        "consensus-threshold", "Number of search result appearances needed to be considered a matching object.", '\0', arrrgh::Optional, 25);
    const auto &outputProgressionFile = parser.add<std::string>(
        "output-progression-file", "Path to a csv file showing scores after every query.", '\0', arrrgh::Optional, "NONE_SELECTED");
    const auto &progressionIterationLimit = parser.add<int>(
        "progression-iteration-limit", "For producing a progression file of a certain length, limit the number of queries processed.", '\0', arrrgh::Optional, -1);
    const auto &subsetStartIndex = parser.add<int>(
        "subset-start-index", "Query index to start from.", '\0', arrrgh::Optional, 0);
    const auto &subsetEndIndex = parser.add<int>(
        "subset-end-index", "Query index to end at. Must be equal or less than the --sample-count parameter.", '\0', arrrgh::Optional, -1);
    const auto &showHelp = parser.add<bool>(
        "help", "Show this help message.", 'h', arrrgh::Optional, false);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if (showHelp.value())
        std::cout << std::endl
                  << "Done." << std::endl;
    {
        return 0;
    }

    std::vector<std::experimental::filesystem::path> queryFiles = ShapeDescriptor::utilities::listDirectory(queryDirectory.value());
    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(signatureDirectory.value());

    SignatureIndex *signatureIndex = readSignatureIndex("output/lsh/index.dat");

    std::vector<std::vector<int>> permutations = signatureIndex->permutations;

    runSignatureQuery(queryFiles.at(0), signatureIndex, supportRadius, haystackFiles);
   
    // --- TODO: Loop and query partial objects -----
    /*
    ObjectQueryResult runSignatureQuery(
        std::experimental::filesystem::path queryFile,
        std::vector<std::vector<int>> permutations,
        float supportRadius,
        size_t seed,
        unsigned int resultsPerQuery,
        unsigned int consensusThreshold,
        std::vector<std::experimental::filesystem::path> &haystackFiles,
        std::string outputProgressionFile,
        int progressionFileIterationLimit)
    */
   // -----------------------------------------------

    std::cout << "Done." << std::endl;
}