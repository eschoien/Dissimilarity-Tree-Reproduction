#include <iostream>
#include <arrrgh.hpp>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <projectSymmetry/descriptors/binaryRICIConverter.h>
#include <projectSymmetry/descriptors/flexibleQUICCISearcher.h>
#include <json.hpp>
#include <json/tsl/ordered_map.h>
#include <git.h>
#include <projectSymmetry/clustering/IndexQueryer.h>
#include <projectSymmetry/clustering/ClusterIO.h>
#include <random>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

struct SearchResult {
    unsigned int imageID;
    unsigned int fileID;
    float executionTimeSeconds;
    ShapeDescriptor::beta::distanceType score;
};

struct QueryDescriptor {
    unsigned short fileID;
    unsigned int imageID;
    unsigned int datasetImageID;
};

struct QueryResult {
    double executionTimeSeconds;
    cluster::path queryFile;
    unsigned int queryFileID;
    unsigned int queryImageID;
    unsigned int queryDatasetImageID;
    unsigned int bestSearchResultFileID;
    unsigned int bestSearchResultImageID;
    unsigned int nodesVisited;
    unsigned int scannedLeafNodeCount;
    float bestMatchScore;
    std::vector<cluster::QueryResult> result;
};

std::string padTo(const std::string &filename, ShapeDescriptor::beta::distanceType score, unsigned int padToSize) {
    std::string scoreString = std::to_string((unsigned long long)(score));
    unsigned int filenameLength = filename.length();
    unsigned int scoreLength = scoreString.length();
    unsigned int totalLength = filenameLength + scoreLength + 2;
    std::ostringstream out;
    out << filename << ": " << scoreString;
    for(unsigned int i = totalLength; i < padToSize; i++) {
        out << " ";
    }
    return out.str();
}

bool searchResultComparator(SearchResult const &lhs, SearchResult const &rhs) {
    if (lhs.score != rhs.score) {
        return lhs.score < rhs.score;
    }
    if (lhs.fileID != rhs.fileID) {
        return lhs.fileID < rhs.fileID;
    }
    return lhs.imageID < rhs.imageID;
}

int main(int argc, const char** argv) {


    arrrgh::parser parser("indexedSearchBenchmark", "Perform a sequential search through a list of descriptors.");
    const auto &indexDirectory = parser.add<std::string>(
            "index-directory", "The directory containing the index to be queried.", '\0',
            arrrgh::Required, "");
    const auto &queryDirectory = parser.add<std::string>(
            "query-directory", "The directory containing query mesh files.", '\0', arrrgh::Required, "");
    const auto &randomSeed = parser.add<size_t>(
            "random-seed", "Random seed to use (allows results to be reproducible).", '\0', arrrgh::Required, -1);
    const auto &count = parser.add<int>(
            "sample-count", "Number of queries to try.", '\0', arrrgh::Required, -1);
    const auto &numberOfSearchResultsToGenerate = parser.add<int>(
            "search-results-per-query", "Number of search results to generate per query.", '\0', arrrgh::Required, 1);
    const auto &forceGPU = parser.add<int>(
            "force-gpu", "Index of the GPU device to use for search kernels.", '\0', arrrgh::Optional, -1);
    const auto &singleQueryIndex = parser.add<int>(
            "single-query-index", "Only execute a single search query from the sequence.", '\0', arrrgh::Optional, -1);
    const auto &outputFile = parser.add<std::string>(
            "output-file", "Path to a JSON file to which to write the search results.", '\0', arrrgh::Optional, "NONE_SELECTED");
    const auto &supportRadius = parser.add<float>(
            "support-radius", "Support radius to use for generating quicci descriptors.", '\0', arrrgh::Optional, 1.0);
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

    cudaDeviceProp gpuProperties = ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());

    std::cout << "Build info: " << GitMetadata::CommitSHA1() << ", by " << GitMetadata::AuthorName() << " on " << GitMetadata::CommitDate() << std::endl;

    std::cout << "Reading cluster file.." << std::endl;
    Cluster* cluster = readCluster(cluster::path(indexDirectory.value()) / "index.dat");
    std::cout << "\tCluster contains " << cluster->nodes.size() << " nodes, " << cluster->images.size() << " images, and indexes " << cluster->indexedFiles.size() << " files." << std::endl;

    std::vector<std::experimental::filesystem::path> queryFiles = ShapeDescriptor::utilities::listDirectory(queryDirectory.value());

    std::cout << "Computing image counts for query files.." << std::endl;
    std::vector<QueryDescriptor> queryImageList;

    std::unordered_map<std::string, size_t> fileImageBaseIndexMap;

    std::vector<ShapeDescriptor::cpu::Mesh> queryMeshes(queryFiles.size());

    unsigned int nextDatasetImageID = 0;
    for(unsigned int fileID = 0; fileID < queryFiles.size(); fileID++) {
        std::cout << "\rProcessing file " << (fileID + 1) << "/" << queryFiles.size() << ".." << std::flush;

        fileImageBaseIndexMap[queryFiles.at(fileID).string()] = nextDatasetImageID;

        // Load meshes
        ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(queryFiles.at(fileID), true);
        ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(
                gpuMesh);

        queryImageList.reserve(queryImageList.size() + descriptorOrigins.length);
        for(unsigned int imageID = 0; imageID < descriptorOrigins.length; imageID++) {
            queryImageList.push_back({(unsigned short) fileID, imageID, nextDatasetImageID});
            nextDatasetImageID++;
        }

        queryMeshes.at(fileID) = mesh;
        //ShapeDescriptor::free::mesh(mesh);
        ShapeDescriptor::free::mesh(gpuMesh);
        ShapeDescriptor::free::array(descriptorOrigins);
    }

    std::cout << std::endl << "Image counting complete. Total query count in dataset: " << queryImageList.size() << std::endl;

    std::minstd_rand0 generator{randomSeed.value()};
    std::shuffle(std::begin(queryImageList), std::end(queryImageList), generator);
    std::vector<QueryDescriptor> trimmedQueryDescriptorList;
    queryImageList.resize(count.value());
    trimmedQueryDescriptorList.swap(queryImageList);

    std::vector<QueryResult> queryResults;

    for(unsigned int i = 0; i < trimmedQueryDescriptorList.size(); i++) {
        if(singleQueryIndex.value() != -1 && singleQueryIndex.value() != i) {
            continue;
        }
        std::cout << "Processing query " << (i + 1) << "/" << trimmedQueryDescriptorList.size() << std::endl;

        QueryDescriptor query = trimmedQueryDescriptorList.at(i);

        ShapeDescriptor::cpu::Mesh mesh = queryMeshes.at(query.fileID); //ShapeDescriptor::utilities::loadMesh(queryFiles.at(query.fileID), true);
        ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(
                gpuMesh);

        // Compute the descriptor(s)
        //std::cout << "Computing descriptors.." << std::endl;
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> riciDescriptors =
                ShapeDescriptor::gpu::generateRadialIntersectionCountImages(gpuMesh, descriptorOrigins, supportRadius.value());

        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = convertRICIToModifiedQUICCI(riciDescriptors);
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> queryDescriptors = ShapeDescriptor::copy::deviceArrayToHost(descriptors);

        cluster::debug::QueryRunInfo info;

        //std::cout << "Querying.." << std::endl;
        ShapeDescriptor::QUICCIDescriptor queryImage = queryDescriptors.content[query.imageID];

        std::vector<cluster::QueryResult> results = cluster::query(cluster, queryImage, numberOfSearchResultsToGenerate.value(), &info);
        std::cout << "\tExecution time: " << info.totalQueryTime << std::endl;

        // Free memory
        ShapeDescriptor::free::array(descriptorOrigins);
        ShapeDescriptor::free::array(descriptors);
        ShapeDescriptor::free::array(queryDescriptors);
        ShapeDescriptor::free::array(riciDescriptors);
        //ShapeDescriptor::free::mesh(mesh);
        ShapeDescriptor::free::mesh(gpuMesh);

        QueryResult result;
        result.executionTimeSeconds = info.totalQueryTime;
        result.queryFile = queryFiles.at(query.fileID);
        result.queryFileID = query.fileID;
        result.queryImageID = query.imageID;
        result.queryDatasetImageID = query.datasetImageID;
        result.bestSearchResultFileID = results.at(0).entry.fileID;
        result.bestSearchResultImageID = results.at(0).entry.imageID;
        result.bestMatchScore = results.at(0).score;
        result.nodesVisited = info.visitedNodeCount;
        result.scannedLeafNodeCount = info.scannedLeafNodeCount;
        result.result = results;

        queryResults.push_back(result);

        if(outputFile.value() != "NONE_SELECTED" && (i % 25 == 24 || singleQueryIndex.value() != -1)) {
            json outJson;

            outJson["version"] = "v8";
            outJson["resultCount"] = numberOfSearchResultsToGenerate.value();
            outJson["queryObjectSupportRadius"] = supportRadius.value();
            outJson["randomSeed"] = randomSeed.value();
            outJson["sampleCount"] = count.value();

            outJson["buildinfo"] = {};
            outJson["buildinfo"]["commit"] = GitMetadata::CommitSHA1();
            outJson["buildinfo"]["commit_author"] = GitMetadata::AuthorName();
            outJson["buildinfo"]["commit_date"] = GitMetadata::CommitDate();

            outJson["gpuInfo"] = {};
            outJson["gpuInfo"]["name"] = gpuProperties.name;
            outJson["gpuInfo"]["clockrate"] = gpuProperties.clockRate;
            outJson["gpuInfo"]["memoryCapacityInMB"] = gpuProperties.totalGlobalMem / (1024 * 1024);

            outJson["cluster"] = {};
            outJson["cluster"]["imageCount"] = cluster->images.size();
            outJson["cluster"]["nodeCount"] = cluster->nodes.size();
            outJson["cluster"]["maxImagesPerLeafNode"] = cluster->maxImagesPerLeafNode;

            outJson["queryDirectory"] = cluster::path(queryDirectory.value()).string();
            outJson["indexDirectory"] = cluster::path(indexDirectory.value()).string();

            outJson["results"] = {};

            for(size_t resultIndex = 0; resultIndex < queryResults.size(); resultIndex++) {
                outJson["results"].emplace_back();
                outJson["results"][resultIndex] = {};
                outJson["results"][resultIndex]["executionTimeSeconds"] = queryResults.at(resultIndex).executionTimeSeconds;
                outJson["results"][resultIndex]["queryFile"] = queryResults.at(resultIndex).queryFile;
                outJson["results"][resultIndex]["queryFileID"] = queryResults.at(resultIndex).queryFileID;
                outJson["results"][resultIndex]["queryImageID"] = queryResults.at(resultIndex).queryImageID;
                outJson["results"][resultIndex]["queryDatasetImageID"] = queryResults.at(resultIndex).queryDatasetImageID;
                outJson["results"][resultIndex]["bestSearchResultImageID"] = queryResults.at(resultIndex).bestSearchResultImageID;
                outJson["results"][resultIndex]["bestSearchResultFileID"] = queryResults.at(resultIndex).bestSearchResultFileID;
                outJson["results"][resultIndex]["bestMatchScore"] = queryResults.at(resultIndex).bestMatchScore;
                outJson["results"][resultIndex]["nodesVisited"] = queryResults.at(resultIndex).nodesVisited;
                outJson["results"][resultIndex]["leafNodesVisited"] = queryResults.at(resultIndex).scannedLeafNodeCount;
                outJson["results"][resultIndex]["searchResults"] = {};
                for(unsigned int i = 0; i < queryResults.at(resultIndex).result.size(); i++) {
                    outJson["results"][resultIndex]["searchResultFileIDs"].emplace_back();
                    outJson["results"][resultIndex]["searchResultFileIDs"][i]["score"] = queryResults.at(resultIndex).result.at(i).score;
                    outJson["results"][resultIndex]["searchResultFileIDs"][i]["imageID"] = queryResults.at(resultIndex).result.at(i).entry.imageID;
                    outJson["results"][resultIndex]["searchResultFileIDs"][i]["fileID"] = queryResults.at(resultIndex).result.at(i).entry.fileID;
                }
            }

            std::ofstream outFile(outputFile.value());
            outFile << outJson.dump(4);
            outFile.close();
        }
    }





    delete cluster;
}