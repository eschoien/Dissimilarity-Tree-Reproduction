#include <vector>
#include <unordered_map>
#include <projectSymmetry/lsh/Hashtable.h>
#include <projectSymmetry/lsh/HashtableIO.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <shapeDescriptor/utilities/fileutils.h>
#include <projectSymmetry/lsh/Signature.h>
#include <projectSymmetry/lsh/Permutation.h>
#include <random>
#include <arrrgh.hpp>
#include <map>
#include <atomic>
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
    if (o1.score == o2.score) {
        return (o1.fileID < o2.fileID);
    }
    return (o1.score > o2.score);
}

// flattens multiple vectors of descriptor entries to one unordered map of descriptor entries and count across the vectors
std::unordered_map<std::string, int> reduce(std::vector<HashtableValue> htValues) {

    std::unordered_map<std::string, int> result;

    for (HashtableValue htv : htValues) {
        for (DescriptorEntry de : htv) {
            result.emplace(descriptorEntryToString(de), 0);
            result[descriptorEntryToString(de)] += 1;
        }
    }
    
    return result;
}

// prints each item in an unordered map of descriptor entry string and int
void printResultEntries(std::unordered_map<std::string, int> result) {
    for (auto const &pair : result) {
        std::cout << "{" << pair.first << ": " << pair.second << "}\n";
    }
}

unsigned int objectIDfromPath(std::string path) {
    return std::stoi(path.substr(path.find("/T")+2));
}

QueryResult runHashtableQuery(
        std::experimental::filesystem::path queryFile,
        std::vector<Hashtable*>& hashtables,
        std::vector<std::vector<int>>& permutations,
        float supportRadius,
        unsigned int descriptorsPerObjectLimit,
        double JACCARD_THRESHOLD,
        size_t seed,
        int k
        ) {

    int numPermutations = 10;
    int numberOfObjects = 383;

      // --- load query object and compute descriptors
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(queryFile, true);
    
    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);
    
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = 
            ShapeDescriptor::utilities::generateSpinOriginBuffer(gpuMesh);
    
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
    std::vector<ObjectScore> objectScores(numberOfObjects, *(new ObjectScore));

    for (uint i = 1; i <= numberOfObjects; i++) {
            objectScores[i-1].fileID = i;
            objectScores[i-1].score = 0;
        }

    uint queryFileID = objectIDfromPath(queryFile.string());

    std::cout << "Partial object: " << queryFileID << std::endl;
    
    std::vector<uint> order(350000); // highest (queryDescriptors.length);
    for(uint s = 0; s < 350000; s++) {
        order.at(s) = s;
    }

    // Comment out line below to disable randomness?
    std::shuffle(order.begin(), order.end(), generator);
    #pragma omp parallel for schedule(dynamic)
    for(uint q = 0; q < descriptorsPerObjectLimit; q++) {

        std::vector<int> queryDescriptorSignature;
        uint queryDescriptorIndex = order[q] % queryDescriptors.length;

        computeDescriptorSignature(queryDescriptors.content[queryDescriptorIndex], &queryDescriptorSignature, permutations);

        // vector of candidate descriptor entry vectors
        std::vector<HashtableValue> htValues(numPermutations);

        for (uint i = 0; i < numPermutations; i++) {
            // get the correct HashtableValue for each hashtable
            htValues[i] = hashtables.at(i)->at(queryDescriptorSignature[i]);
        }

        std::unordered_map<std::string, int> result = reduce(htValues);

        /*
        std::cout << "\nResult entries:\n" << std::endl;

        for (auto const &pair : result) {
            if (pair.second >= JACCARD_THRESHOLD)
                std::cout << "{" << pair.first << ": " << pair.second << "}\n";
        }
        */

        std::vector<int> matches(numberOfObjects, 0);

        for (auto const &pair : result) {
            
            // sufficient match
            if (((double) pair.second / (double) numPermutations) > JACCARD_THRESHOLD - 0.000001) {
                // std::cout << "de count " << pair.second << " : JT: " << JACCARD_THRESHOLD - 0.000001 << std::endl;
                // remove the need for stoi here at some point
                unsigned int oID = std::stoi((pair.first).substr(0, (pair.first).find("-")));
                // std::cout << "object id " << oID << std::endl;
                matches[oID-1] = 1;
            }
        }

        for (uint i = 0; i < matches.size(); i++) {

            if (matches[i] == 1) {
                objectScores[i].score++;
            }
        }

        // std::cout << "Matches descriptor " << q << " : " << std::accumulate(matches.begin(), matches.end(), 0) << std::endl;
    }

    std::vector<ObjectScore> bestMatches;
    std::copy(objectScores.begin(), objectScores.end(), std::back_inserter(bestMatches));
    std::sort(bestMatches.begin(), bestMatches.end(), compareScores);


    /*
    for (int k = 0; k < 10; k++) {
        ObjectScore os = bestMatches[k];
        std::cout << os.fileID << " : " << os.score << std::endl;
    }
    */

    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - queryStartTime);
    std::cout << "\tExecution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    ShapeDescriptor::free::array(descriptorOrigins);
    ShapeDescriptor::free::array(descriptors);
    ShapeDescriptor::free::array(queryDescriptors);
    ShapeDescriptor::free::array(riciDescriptors);
    ShapeDescriptor::free::mesh(mesh);
    ShapeDescriptor::free::mesh(gpuMesh);
    
    
    QueryResult result;
    result.queryFileID = queryFileID;
    result.queryFileScore = objectScores[queryFileID-1].score;
    result.kLimit = k;
    for (int i = 0; i < k; i++) {
        result.bestMatches.push_back(bestMatches[i].fileID);
        result.bestMatchScores.push_back(objectScores[bestMatches[i].fileID-1].score);
    }
    result.executionTimeSeconds = float(duration.count()) / 1000.0f;

    return result;

}

int main(int argc, const char **argv) {
    arrrgh::parser parser("hashtableSearcher", "Search for similar objects using LSH including hashtables.");
    const auto &hashtableDirectory = parser.add<std::string>(
        "hashtable-directory", "The directory containing hashtables to query.", '\0', arrrgh::Required, "output/lsh/hashtables/");
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
    const auto &numberOfPermutations = parser.add<int>(
        "numPermutations", "Number of permutations", '\0', arrrgh::Optional, 10);

    try {
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
    std::vector<std::experimental::filesystem::path> hashtableFiles = ShapeDescriptor::utilities::listDirectory(hashtableDirectory.value());

    std::vector<Hashtable*> hashtables(numberOfPermutations.value());

    for (int i = 0; i < numberOfPermutations.value(); i++) {
        std::cout << hashtableFiles[i].string() << std::endl;
        hashtables[i] = readHashtable("output/lsh/hashtables/H"+std::to_string(i)+".dat");
    }

    std::vector<std::vector<int>> permutations = create_permutations(numberOfPermutations.value(), (size_t) 32895532);

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::vector<QueryResult> searchResults;

    unsigned int startIndex = subsetStartIndex.value();
    unsigned int endIndex = subsetEndIndex.value() != -1 ? subsetEndIndex.value() : queryFiles.size();
    for(unsigned int queryFile = startIndex; queryFile < endIndex; queryFile++) {
        std::cout << "Processing query " << (queryFile + 1) << "/" << endIndex << ": " << queryFiles.at(queryFile).string() << std::endl;
        QueryResult queryResult = runHashtableQuery(queryFiles.at(queryFile), hashtables, permutations, supportRadius.value(), descriptorsPerObjectLimit.value(), JACCARD_THRESHOLD.value(), seed.value(), k.value());
        searchResults.push_back(queryResult);

        if(outputFile.value() != "NONE_SELECTED") {
            json outJson;

            outJson["version"] = version.value();
            outJson["queryObjectSupportRadius"] = supportRadius.value();

            outJson["queryDirectory"] = cluster::path(queryDirectory.value()).string();
            outJson["hashtables"] = hashtables.size();
            outJson["dumpFilePath"] = cluster::path(outputFile.value()).string();
            outJson["randomSeed"] = seed.value();
            outJson["queryStartIndex"] = startIndex;
            outJson["queryEndIndex"] = endIndex;
            outJson["descriptorsPerObjectLimit"] = descriptorsPerObjectLimit.value();
            outJson["JACCARD_THRESHOLD"] = JACCARD_THRESHOLD.value() - 0.000001;
            outJson["permutations"] = permutations.size();

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

    // Measure total execution time
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl << "Hashtable search complete. " << std::endl;
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;
    
    // delete signatureIndex;

    std::cout << "Done." << std::endl;

   return 0;
}