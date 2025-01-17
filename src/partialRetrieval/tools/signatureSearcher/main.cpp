#include <arrrgh.hpp>
#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <atomic>
#include <string>
#include <random>
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
#include <json.hpp>
#include <json/tsl/ordered_map.h>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;
struct ObjectScore {
    unsigned int fileID;
    unsigned int score;
};
struct QueryResult {
    unsigned int queryFileID;
    unsigned int queryFileScore;
    std::vector<unsigned int> bestMatches;
    std::vector<unsigned int> bestMatchScores;
    float executionTimeSeconds;
    unsigned int kLimit;
};


bool compareScores(ObjectScore o1, ObjectScore o2) {
    /*if (o1.score == o2.score) {
        return (o1.fileID < o2.fileID);
    }*/
    return (o1.score > o2.score);
}

QueryResult runSignatureQuery(
        std::experimental::filesystem::path queryFile,
        SignatureIndex *signatureIndex,
        float supportRadius,
        unsigned int descriptorsPerObjectLimit,
        double JACCARD_THRESHOLD,
        size_t seed,
        int k
        ) {

  // --- load partial object and compute descriptors
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(queryFile, true);
    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = 
            ShapeDescriptor::utilities::generateSpinOriginBuffer(gpuMesh);

    // Compute the descriptor(s)
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> riciDescriptors =
            ShapeDescriptor::gpu::generateRadialIntersectionCountImages(gpuMesh, descriptorOrigins, supportRadius);

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = convertRICIToModifiedQUICCI(riciDescriptors);
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> queryDescriptors = ShapeDescriptor::copy::deviceArrayToHost(descriptors);
    // ----

    std::random_device rd("/dev/urandom");
    size_t randomSeed = seed != 0 ? seed : rd();
    std::minstd_rand0 generator{randomSeed};
    std::uniform_real_distribution<float> distribution(0, 1);
    
    
    std::chrono::steady_clock::time_point queryStartTime = std::chrono::steady_clock::now();
    std::vector<ObjectScore> objectScores(signatureIndex->objectCount, *(new ObjectScore));
   
    // Create partial object signatures
    ObjectSignature* queryObjectSignature = new ObjectSignature;
    
    std::string path_string = queryFile.string();
    std::size_t pos = path_string.find("/T");
    queryObjectSignature->file_id = std::stoi(path_string.substr(pos+2));
    unsigned int fileID = queryObjectSignature->file_id;

    std::cout << "Partial object: " << queryObjectSignature->file_id << std::endl;
    
    std::vector<unsigned int> order(350000); // (queryDescriptors.length);
    for(unsigned int s = 0; s < 350000; s++) {
        order.at(s) = s;
    }
    // Comment out line below to disable randomness?
    std::shuffle(order.begin(), order.end(), generator);

    for(unsigned int i = 0; i < descriptorsPerObjectLimit; i++) {
        DescriptorSignature descriptorSignature;
        descriptorSignature.descriptor_id = order[i] % queryDescriptors.length + 1; //i + 1;
        computeDescriptorSignature(queryDescriptors.content[order[i] % queryDescriptors.length], &(descriptorSignature.signatures), signatureIndex->permutations);
        queryObjectSignature->descriptorSignatures.push_back(descriptorSignature);
    }

    // parallize this
    // Loop through commplete object signature files
    #pragma omp parallel for schedule(dynamic)
    for(unsigned int i = 0; i < signatureIndex->objectCount; i++) {
        // ObjectSignature* objectSignature = readSignature(haystackFiles.at(i), signatureIndex->numPermutations);
        ObjectSignature* objectSignature = &(signatureIndex->objectSignatures[i]);
        objectScores[objectSignature->file_id-1].fileID = objectSignature->file_id;
        objectScores[objectSignature->file_id-1].score = 0;

        // Loop through complete object descripor signatures 
        // Comment out line below to disable randomness?
        for(unsigned int k = 0; k < queryObjectSignature->descriptorSignatures.size(); k++) {

            std::vector<int> querySignature = queryObjectSignature->descriptorSignatures[k].signatures;
            
            for (unsigned int j = 0; j < descriptorsPerObjectLimit; j++) {
                std::vector<int> candidateSignature = objectSignature->descriptorSignatures[j].signatures;
                //std::vector<int> candidateSignature = objectSignature.descriptorSignatures[j].signatures;

                double jaccardSimilarity = computeJaccardSimilarity(querySignature, candidateSignature);
                
                if (jaccardSimilarity >= JACCARD_THRESHOLD - 0.000001) {
                    objectScores[objectSignature->file_id-1].score++;
                    break;
                }

            }
        }
        // delete objectSignature;
    }   
    delete queryObjectSignature;

    // best matching object
    // returns the object with the highest number of descriptor signatures with jaccard similarity greater than threshold
    std::vector<ObjectScore> bestMatches;
    std::copy(objectScores.begin(), objectScores.end(), std::back_inserter(bestMatches));
    std::sort(bestMatches.begin(), bestMatches.end(), compareScores);
    // unsigned int bestMatch = std::distance(objectScores.begin(), std::max_element(objectScores.begin(), objectScores.end())) + 1;
    // std::cout << "Best matching object " << bestMatch << std::endl;

    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - queryStartTime);
    std::cout << "\tTotal execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;
    std::cout << std::endl;

    ShapeDescriptor::free::array(descriptorOrigins);
    ShapeDescriptor::free::array(descriptors);
    ShapeDescriptor::free::array(queryDescriptors);
    ShapeDescriptor::free::array(riciDescriptors);
    ShapeDescriptor::free::mesh(mesh);
    ShapeDescriptor::free::mesh(gpuMesh);
    
    QueryResult result;
    result.queryFileID = fileID;
    result.queryFileScore = objectScores[fileID-1].score;
    result.kLimit = k;
    for (int i = 0; i < k; i++) {
        result.bestMatches.push_back(bestMatches[i].fileID);
        result.bestMatchScores.push_back(objectScores[bestMatches[i].fileID-1].score);
    }
    result.executionTimeSeconds = float(duration.count()) / 1000.0f;

    return result;
}

// modified from another object search, needs further changes
int main(int argc, const char **argv) {
    arrrgh::parser parser("signatureSearcher", "Search for similar objects using LSH.");
    const auto &signatureFile = parser.add<std::string>(
        "signature-file", "The signature file to be queried.", '\0', arrrgh::Required, "output/lsh/index.dat");
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
    const auto &k = parser.add<int>(
        "k", "k-nearest neighbour", '\0', arrrgh::Optional, 5);
    const auto &version = parser.add<std::string>(
        "version", "version of the signature searcher", '\0', arrrgh::Optional, "v1");
    const auto &showHelp = parser.add<bool>(
        "help", "Show this help message.", 'h', arrrgh::Optional, false);
    const auto &descriptorsPerObjectLimit = parser.add<int>(
        "descriptorsPerObjectLimit", "descriptorsPerObjectLimit", '\0', arrrgh::Optional, 200);
    const auto &JACCARD_THRESHOLD = parser.add<float>(
        "JACCARD_THRESHOLD", "JACCARD_THRESHOLD", '\0', arrrgh::Optional, 0.5);

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
    if (showHelp.value()) {
        return 0;
    }
    
    std::vector<std::experimental::filesystem::path> queryFiles = ShapeDescriptor::utilities::listDirectory(queryDirectory.value());
    // std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(signatureDirectory.value());

    SignatureIndex *signatureIndex = readSignatureIndex(signatureFile.value());

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::vector<QueryResult> searchResults;

    unsigned int startIndex = subsetStartIndex.value();
    unsigned int endIndex = subsetEndIndex.value() != -1 ? subsetEndIndex.value() : queryFiles.size();
    for(unsigned int queryFile = startIndex; queryFile < endIndex; queryFile++) {
        std::cout << "Processing query " << (queryFile + 1) << "/" << endIndex << ": " << queryFiles.at(queryFile).string() << std::endl;
        QueryResult queryResult = runSignatureQuery(queryFiles.at(queryFile), signatureIndex, supportRadius.value(), descriptorsPerObjectLimit.value(), JACCARD_THRESHOLD.value(), seed.value(), k.value());
        searchResults.push_back(queryResult);

        if(outputFile.value() != "NONE_SELECTED") {
            json outJson;

            outJson["version"] = version.value();
            outJson["queryObjectSupportRadius"] = supportRadius.value();

            outJson["queryDirectory"] = cluster::path(queryDirectory.value()).string();
            outJson["signatureFile"] = cluster::path(signatureFile.value()).string();
            outJson["dumpFilePath"] = cluster::path(outputFile.value()).string();
            outJson["randomSeed"] = seed.value();
            outJson["queryStartIndex"] = startIndex;
            outJson["queryEndIndex"] = endIndex;
            outJson["descriptorsPerObjectLimit"] = descriptorsPerObjectLimit.value();
            outJson["JACCARD_THRESHOLD"] = JACCARD_THRESHOLD.value() - 0.000001;
            outJson["permutations"] = signatureIndex->numPermutations;

            outJson["results"] = {};

            for(size_t resultIndex = 0; resultIndex < searchResults.size(); resultIndex++) {
                outJson["results"].emplace_back();
                outJson["results"][resultIndex] = {};
                outJson["results"][resultIndex]["kLimit"] = searchResults.at(resultIndex).kLimit;
                outJson["results"][resultIndex]["queryFileID"] = searchResults.at(resultIndex).queryFileID;
                outJson["results"][resultIndex]["queryFileScore"] = searchResults.at(resultIndex).queryFileScore;
                outJson["bestMatches"] = {};
                outJson["bestMatchScores"] = {};
                for (size_t bestMatchID = 0; bestMatchID < searchResults.at(resultIndex).kLimit; bestMatchID++) {
                    outJson["results"][resultIndex]["bestMatches"][bestMatchID] = searchResults.at(resultIndex).bestMatches.at(bestMatchID);
                    outJson["results"][resultIndex]["bestMatchScores"][bestMatchID] = searchResults.at(resultIndex).bestMatchScores.at(bestMatchID);
                }
                // outJson["results"][resultIndex]["bestMatchScore"] = searchResults.at(resultIndex).bestMatchScore;
                outJson["results"][resultIndex]["executionTimeSeconds"] = searchResults.at(resultIndex).executionTimeSeconds;
            }

            std::ofstream outFile(outputFile.value());
            outFile << outJson.dump(4);
            outFile.close();
        }
    }
    // runSignatureQuery(queryFiles.at(0), signatureIndex, supportRadius, haystackFiles);
    
    // Measure total execution time
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl << "MinHash signature search complete. " << std::endl;
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;
    
    delete signatureIndex;

    std::cout << "Done." << std::endl;
}