#include <iostream>
#include <arrrgh.hpp>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <projectSymmetry/descriptors/binaryRICIConverter.h>
#include <projectSymmetry/clustering/IndexQueryer.h>
#include <projectSymmetry/clustering/ClusterIO.h>
#include <json.hpp>
#include <json/tsl/ordered_map.h>
#include <git.h>
#include <random>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

struct ObjectSearchResult {
    unsigned int fileID;
    unsigned int appearanceCount;
    double score;

    ObjectSearchResult(unsigned int fileID, unsigned int appearanceCount, double score)
        : fileID(fileID), appearanceCount(appearanceCount), score(score) {}
};

struct ObjectQueryResult {
    std::vector<ObjectSearchResult> searchResults;
    std::vector<unsigned int> allOccurrenceCounts;
    std::vector<double> allTotalDistances;
    float executionTimeSeconds;
};

bool searchResultComparator(ObjectSearchResult const &lhs, ObjectSearchResult const &rhs) {
    if (lhs.score != rhs.score) {
        return lhs.score < rhs.score;
    }
    return lhs.fileID < rhs.fileID;
}

ObjectQueryResult runObjectQuery(
        cluster::path queryFile,
        Cluster* cluster,
        float supportRadius,
        size_t seed,
        unsigned int resultsPerQuery,
        unsigned int consensusThreshold,
        std::vector<std::experimental::filesystem::path> &haystackFiles,
        std::string outputProgressionFile,
        int progressionFileIterationLimit) {

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(queryFile, true);
    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(
            gpuMesh);

    // Compute the descriptor(s)
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> riciDescriptors =
            ShapeDescriptor::gpu::generateRadialIntersectionCountImages(gpuMesh, descriptorOrigins, supportRadius);

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = convertRICIToModifiedQUICCI(riciDescriptors);
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> queryDescriptors = ShapeDescriptor::copy::deviceArrayToHost(descriptors);

    std::random_device rd("/dev/urandom");
    size_t randomSeed = seed != 0 ? seed : rd();
    std::minstd_rand0 generator{randomSeed};
    std::uniform_real_distribution<float> distribution(0, 1);

    unsigned int processedCount = 0;
    unsigned int nextResultIndexToComplete = 0;

    std::vector<unsigned int> vertexOrder(queryDescriptors.length);
    for(unsigned int i = 0; i < queryDescriptors.length; i++) {
        vertexOrder.at(i) = i;
    }
    std::shuffle(vertexOrder.begin(), vertexOrder.end(), generator);

    //std::vector<std::vector<ObjectSearchResult>> resultSnapshots;
    //resultSnapshots.reserve(vertexOrder.size());

    std::vector<std::vector<cluster::QueryResult>> allSearchResults(queryDescriptors.length);

    std::vector<unsigned int> searchResultOccurrenceCounts(haystackFiles.size());
    std::vector<double> searchResultTotalDistance(haystackFiles.size());
    std::vector<ObjectSearchResult> searchResults;
    searchResults.reserve(100);

    bool hasSurpassedThreshold = false;

    std::vector<std::vector<unsigned int>> progressionCounts;
    bool enableProgressionDump = outputProgressionFile != "NONE_SELECTED";
    if(enableProgressionDump) {
        consensusThreshold = INT_MAX;
        std::cout << "Number of query object descriptors: " << queryDescriptors.length << std::endl;
    }
    unsigned int nextThresholdToDump = progressionFileIterationLimit == -1 ? 1000 : std::min<unsigned int>(1000, progressionFileIterationLimit);

    std::chrono::steady_clock::time_point queryStartTime = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(dynamic)
    for (unsigned int queryImageIndex = 0; queryImageIndex < queryDescriptors.length; queryImageIndex++) {
        // Basically ends the loop when early exit condition is reached
        if(hasSurpassedThreshold) {
            continue;
        }

        cluster::debug::QueryRunInfo info;
        ShapeDescriptor::QUICCIDescriptor queryImage = queryDescriptors.content[vertexOrder.at(queryImageIndex)];
        std::vector<cluster::QueryResult> queryResults = cluster::query(cluster, queryImage, resultsPerQuery, &info);

#pragma omp critical
        {
            // Ensures search is effectively stopped entirely after a set of search results has been found
            if(!hasSurpassedThreshold) {
                processedCount++;
                if(enableProgressionDump) {
                    std::cout << "\rProcessed: " << processedCount << ", " << nextResultIndexToComplete << std::flush;
                }

                allSearchResults.at(queryImageIndex) = queryResults;

                // Only recount if it advances the list
                if(queryImageIndex == nextResultIndexToComplete) {
                    // Reset count arrays
                    std::fill(searchResultOccurrenceCounts.begin(), searchResultOccurrenceCounts.end(), 0);
                    std::fill(searchResultTotalDistance.begin(), searchResultTotalDistance.end(), 0);

                    if(enableProgressionDump) {
                        progressionCounts.resize(0);
                    }

                    // Counting occurrences up to this point
                    for(unsigned resultIndex = 0; resultIndex < allSearchResults.size(); resultIndex++) {
                        const std::vector<cluster::QueryResult> &results = allSearchResults.at(resultIndex);
                        // Don't look at search results that may have ended further up ahead.
                        // This ensures that even though some speed is sacrificed, the program always produces consistent results.
                        if(results.size() == 0 || hasSurpassedThreshold) {
                            break;
                        }

                        nextResultIndexToComplete = resultIndex + 1;

                        // Each search result votes for one object it thinks fits best
                        const cluster::QueryResult &result = results.at(0);
                        searchResultOccurrenceCounts.at(result.entry.fileID)++;
                        searchResultTotalDistance.at(result.entry.fileID) += result.score;

                        // also needed for guaranteeing consistent results.
                        // assumes single object returned as best match though
                        // TODO: greater than one search result will cause a result to be included more than once
                        if (searchResultOccurrenceCounts.at(result.entry.fileID) == consensusThreshold) {
                            std::cout << "\tFound match: "
                                      << haystackFiles.at(result.entry.fileID).filename().string() << ", "
                                      << searchResultOccurrenceCounts.at(result.entry.fileID) << ", "
                                      << searchResultTotalDistance.at(result.entry.fileID) << std::endl;
                            searchResults.emplace_back(result.entry.fileID,
                                                       searchResultOccurrenceCounts.at(result.entry.fileID),
                                                       searchResultTotalDistance.at(result.entry.fileID));

                            hasSurpassedThreshold = true;
                            break;
                        }

                        if(enableProgressionDump) {
                            progressionCounts.push_back(searchResultOccurrenceCounts);

                            if((progressionCounts.size() >= nextThresholdToDump) || progressionCounts.size() == queryDescriptors.length) {
                                std::cout << "Writing " << progressionCounts.size() << " intermediates.." << std::endl;

                                std::ofstream progressionFileStream(outputProgressionFile, std::ios::out);
                                for(unsigned int i = 0; i < haystackFiles.size(); i++) {
                                    const cluster::filesystem::path &haystackFile = haystackFiles.at(i);
                                    progressionFileStream << haystackFile.filename() << (i + 1 == haystackFiles.size() ? "\n" : ", ");
                                }
                                unsigned int limit = progressionCounts.size();
                                if(limit > progressionFileIterationLimit && progressionFileIterationLimit != -1) {
                                    limit = progressionFileIterationLimit;
                                }
                                for(unsigned int j = 0; j < limit; j++) {
                                    for(unsigned int i = 0; i < haystackFiles.size(); i++) {
                                        progressionFileStream << progressionCounts.at(j).at(i) << (i + 1 == haystackFiles.size() ? "\n" : ", ");
                                    }
                                }
                                progressionFileStream.close();

                                if(progressionCounts.size() >= progressionFileIterationLimit && progressionFileIterationLimit != -1) {
                                    exit(0);
                                }

                                nextThresholdToDump += 1000;
                                nextThresholdToDump = std::min<unsigned int>(nextThresholdToDump, progressionFileIterationLimit);
                            }
                        }
                    }
                }
            };
        }
    }

    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - queryStartTime);
    std::cout << "\tTotal execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;


    // Free memory
    ShapeDescriptor::free::array(descriptorOrigins);
    ShapeDescriptor::free::array(descriptors);
    ShapeDescriptor::free::array(queryDescriptors);
    ShapeDescriptor::free::array(riciDescriptors);
    ShapeDescriptor::free::mesh(mesh);
    ShapeDescriptor::free::mesh(gpuMesh);


    ObjectQueryResult result;
    result.searchResults = searchResults;
    result.executionTimeSeconds = float(duration.count()) / 1000.0f;
    result.allOccurrenceCounts = searchResultOccurrenceCounts;
    result.allTotalDistances = searchResultTotalDistance;

    return result;
}

int main(int argc, const char** argv) {
    arrrgh::parser parser("objectSearch", "Search for a partial query object in an index.");
    const auto &indexDirectory = parser.add<std::string>(
            "index-directory", "The directory containing the index to be queried.", '\0',
            arrrgh::Required, "");
    const auto &haystackDirectory = parser.add<std::string>(
            "haystack-directory", "The directory containing objects that have been indexed.", '\0',
            arrrgh::Required, "");
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
    //const auto &matchingObjectCount = parser.add<int>(
    //        "object-threshold", "Number of search result objects which surpass the consensus threshold to find.", '\0', arrrgh::Optional, 1);
    const auto &forceGPU = parser.add<int>(
            "force-gpu", "Index of the GPU device to use for search kernels.", '\0', arrrgh::Optional, -1);
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

    try {
        parser.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if (showHelp.value()) {
        return 0;
    }


    cudaDeviceProp gpuProperties;
    if(forceGPU.value() != -1) {
        gpuProperties = ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());
    }

    std::cout << "Build info: " << GitMetadata::CommitSHA1() << ", by " << GitMetadata::AuthorName() << " on " << GitMetadata::CommitDate() << std::endl;

    std::vector<std::experimental::filesystem::path> queryFiles = ShapeDescriptor::utilities::listDirectory(queryDirectory.value());
    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(haystackDirectory.value());

    std::cout << "Reading cluster file.." << std::endl;
    Cluster* cluster = readCluster(cluster::path(indexDirectory.value()) / "index.dat");
    std::cout << "\tCluster contains " << cluster->nodes.size() << " nodes and " << cluster->images.size() << " images." << std::endl;
    std::cout << "\tIt was built using commit " << cluster->indexFileCreationCommitHash << std::endl;

    std::vector<ObjectQueryResult> searchResults;

    unsigned int startIndex = subsetStartIndex.value();
    unsigned int endIndex = subsetEndIndex.value() != -1 ? subsetEndIndex.value() : queryFiles.size();
    for(unsigned int queryFile = startIndex; queryFile < endIndex; queryFile++) {
        std::cout << "Processing query " << (queryFile + 1) << "/" << endIndex << ": " << queryFiles.at(queryFile).string() << std::endl;
        ObjectQueryResult queryResult = runObjectQuery(
                queryFiles.at(queryFile), cluster, supportRadius.value(), seed.value(),
                resultsPerQuery.value(), consensusThreshold.value(), haystackFiles, outputProgressionFile.value(), progressionIterationLimit.value());
        searchResults.push_back(queryResult);

        if(outputFile.value() != "NONE_SELECTED") {
            json outJson;

            outJson["version"] = "v8";
            outJson["resultCount"] = resultsPerQuery.value();
            outJson["queryObjectSupportRadius"] = supportRadius.value();

            outJson["buildinfo"] = {};
            outJson["buildinfo"]["commit"] = GitMetadata::CommitSHA1();
            outJson["buildinfo"]["commit_author"] = GitMetadata::AuthorName();
            outJson["buildinfo"]["commit_date"] = GitMetadata::CommitDate();

            outJson["cluster"] = {};
            outJson["cluster"]["imageCount"] = cluster->images.size();
            outJson["cluster"]["nodeCount"] = cluster->nodes.size();
            outJson["cluster"]["maxImagesPerLeafNode"] = cluster->maxImagesPerLeafNode;

            outJson["queryDirectory"] = cluster::path(queryDirectory.value()).string();
            outJson["haystackDirectory"] = cluster::path(haystackDirectory.value()).string();
            outJson["indexDirectory"] = cluster::path(indexDirectory.value()).string();
            outJson["resultsPerQuery"] = resultsPerQuery.value();
            outJson["dumpFilePath"] = cluster::path(outputFile.value()).string();
            outJson["randomSeed"] = seed.value();
            outJson["consensusThreshold"] = consensusThreshold.value();
            outJson["queryStartIndex"] = startIndex;
            outJson["queryEndIndex"] = endIndex;

            outJson["results"] = {};

            for(size_t resultIndex = 0; resultIndex < searchResults.size(); resultIndex++) {
                outJson["results"].emplace_back();
                outJson["results"][resultIndex] = {};
                outJson["results"][resultIndex]["executionTimeSeconds"] = searchResults.at(resultIndex).executionTimeSeconds;
                outJson["results"][resultIndex]["allTotalDistances"] = searchResults.at(resultIndex).allTotalDistances;
                outJson["results"][resultIndex]["allOccurrenceCounts"] = searchResults.at(resultIndex).allOccurrenceCounts;
                outJson["results"][resultIndex]["queryFile"] = queryFiles.at(startIndex + resultIndex).string();
                outJson["results"][resultIndex]["searchResults"] = {};
                for(unsigned int i = 0; i < searchResults.at(resultIndex).searchResults.size(); i++) {
                    outJson["results"][resultIndex]["searchResults"].emplace_back();
                    outJson["results"][resultIndex]["searchResults"][i]["score"] = searchResults.at(resultIndex).searchResults.at(i).score;
                    outJson["results"][resultIndex]["searchResults"][i]["objectID"] = searchResults.at(resultIndex).searchResults.at(i).fileID;
                    outJson["results"][resultIndex]["searchResults"][i]["objectFilePath"] = haystackFiles.at(searchResults.at(resultIndex).searchResults.at(i).fileID).string();
                    outJson["results"][resultIndex]["searchResults"][i]["appearanceCount"] = searchResults.at(resultIndex).searchResults.at(i).appearanceCount;
                }
            }

            std::ofstream outFile(outputFile.value());
            outFile << outJson.dump(4);
            outFile.close();
        }

        if(outputProgressionFile.value() != "NONE_SELECTED") {
            // Only process one file if a progression file is generated.
            break;
        }
    }

    delete cluster;
}