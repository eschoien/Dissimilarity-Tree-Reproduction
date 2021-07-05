#include <iostream>
#include <arrrgh.hpp>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageSearcher.cuh>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/read/QUICCIDescriptors.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <projectSymmetry/descriptors/binaryRICIConverter.h>
#include <projectSymmetry/descriptors/flexibleQUICCISearcher.h>
#include <json.hpp>
#include <json/tsl/ordered_map.h>
#include <git.h>
#include <projectSymmetry/descriptors/areaCalculator.h>
#include <projectSymmetry/types/filesystem.h>

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
    double score;
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
    arrrgh::parser parser("simplesearch", "Create partial query meshes from an object set set.");
    const auto &haystackDirectory = parser.add<std::string>(
            "haystack-directory", "The directory containing descriptors which should be queried.", '\0',
            arrrgh::Required, "");
    const auto &queryMesh = parser.add<std::string>(
            "query-mesh", "The mesh which should be found in the haystack objects.", '\0', arrrgh::Required, "");
    const auto &disableModifiedQUICCI = parser.add<bool>(
            "disable-modified-quicci", "By default, the search generates QUICCI query descriptors using the proposed modification for partial retrieval. Disabling this will use the original algorithm.", '\0', arrrgh::Optional, "");
    const auto &forceGPU = parser.add<int>(
            "force-gpu", "Index of the GPU device to use for search kernels.", '\0', arrrgh::Optional, -1);
    const auto &outputFile = parser.add<std::string>(
            "output-file", "Path to a CSV file to which to write the search results.", '\0', arrrgh::Optional, "NONE_SELECTED");


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

    if(forceGPU.value() != -1) {
        ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());
    }

    std::cout << "Build info: " << GitMetadata::CommitSHA1() << ", by " << GitMetadata::AuthorName() << " on " << GitMetadata::CommitDate() << std::endl;

    // Load meshes
    //std::cout << "Loading meshes.." << std::endl;
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(queryMesh.value(), true);


    //ShapeDescriptor::cpu::Mesh impostorVerticesMesh = ShapeDescriptor::utilities::loadMesh("/mnt/NEXUS/datasets/SHREC2016_generated_queries_original/T37 .obj", true);

    // Store them on the GPU
    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    //ShapeDescriptor::gpu::Mesh inpostorGpuMesh = ShapeDescriptor::copy::hostMeshToDevice(impostorVerticesMesh);


    // libShapeDescriptor generators require a list of oriented points from which to compute descriptors
    // You can define these manually, or just grab all vertices from the input object
    // The latter is what the line below does.
    std::cout << "Computing descriptor origins.." << std::endl;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(
            gpuMesh);

    std::cout << "\tMesh loaded, contains " << descriptorOrigins.length << " vertices" << std::endl;

    // Compute the descriptor(s)
    std::cout << "Computing descriptors.." << std::endl;
    float supportRadius = 100.0;

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors;

    if(disableModifiedQUICCI.value()) {
        descriptors = ShapeDescriptor::gpu::generateQUICCImages(gpuMesh, descriptorOrigins, supportRadius);
    } else {
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> riciDescriptors =
                ShapeDescriptor::gpu::generateRadialIntersectionCountImages(gpuMesh, descriptorOrigins, supportRadius);
        descriptors = convertRICIToModifiedQUICCI(riciDescriptors);
        ShapeDescriptor::free::array(riciDescriptors);
    }

    //ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> tempDescriptors = ShapeDescriptor::copy::deviceArrayToHost(descriptors);
    //tempDescriptors.length = 5000;
    //ShapeDescriptor::dump::descriptors(tempDescriptors, "quicci_twointersection.png");

    // Do something with descriptors here
    std::cout << "Computing search results.." << std::endl;
    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(
            haystackDirectory.value());

    std::vector<SearchResult> fileRanking;
    fileRanking.resize(haystackFiles.size());

    SearchResult emptyResult{0, 0, 0,0};
    std::fill(fileRanking.begin(), fileRanking.end(), emptyResult);

    for(unsigned int fileIndex = 0; fileIndex < fileRanking.size(); fileIndex++) {
        fileRanking.at(fileIndex).fileID = fileIndex;
    }

    for (unsigned int i = 0; i < haystackFiles.size(); i++) {
        std::cout << "Processing " << (i + 1) << "/" << haystackFiles.size() << ": " << haystackFiles.at(i)
                  << std::endl;

        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptorsCPU = ShapeDescriptor::read::QUICCIDescriptors(
                haystackFiles.at(i), 8);

        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors = ShapeDescriptor::copy::hostArrayToDevice(
                haystackDescriptorsCPU);

        ShapeDescriptor::free::array(haystackDescriptorsCPU);

        float executionTimeSeconds = 0;

        ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<ShapeDescriptor::beta::distanceType>> searchResults =
                ShapeDescriptor::beta::runFlexibleQUICCISearch(descriptors, haystackDescriptors, &executionTimeSeconds);

        double totalFileDistance = 0;

        for (unsigned int searchResultSet = 0; searchResultSet < searchResults.length; searchResultSet++) {
            //if(searchResults.content[searchResultSet].scores[0] > 10) {
            //    totalFileDistance++;
            //}

            totalFileDistance += searchResults.content[searchResultSet].scores[0];
        }
        std::cout << "\tScore: " << totalFileDistance << std::endl;

        ShapeDescriptor::free::array(haystackDescriptors);
        ShapeDescriptor::free::array(searchResults);

        fileRanking.at(i).score = totalFileDistance;
        fileRanking.at(i).executionTimeSeconds = executionTimeSeconds;

        std::vector<SearchResult> fileRankingListForDisplay = fileRanking;

        std::sort(fileRankingListForDisplay.begin(), fileRankingListForDisplay.end(), searchResultComparator);

        unsigned int startIndex = 0;
        while(fileRankingListForDisplay.at(startIndex).score == 0) {
            startIndex++;
        }

        std::cout << "\tRankings: " << std::endl << "\t";
        for(unsigned int j = startIndex; j < std::min<unsigned int>(startIndex + 50, fileRankingListForDisplay.size()); j++) {
            std::cout << padTo(haystackFiles.at(fileRankingListForDisplay.at(j).fileID).filename().string(), fileRankingListForDisplay.at(j).score, 20)
                      << ((j - startIndex) % 10 == 9 ? "\r\n\t" : "");
        }
        std::cout << std::endl;
    }

    std::sort(fileRanking.begin(), fileRanking.end(), searchResultComparator);

    std::cout << std::endl << "Final rankings: " << std::endl << "\t";
    for(unsigned int j = 0; j < fileRanking.size(); j++) {
        std::cout << padTo(haystackFiles.at(fileRanking.at(j).fileID).filename().string(), fileRanking.at(j).score, 20)
                  << (j % 10 == 9 ? "\r\n\t" : "");
    }
    std::cout << std::endl;

    if(outputFile.value() != "NONE_SELECTED") {
        json outJson;

        outJson["version"] = "v6";
        outJson["resultCount"] = fileRanking.size();
        outJson["queryArea"] = computeMeshArea(mesh);
        outJson["buildinfo"] = {};
        outJson["buildinfo"]["commit"] = GitMetadata::CommitSHA1();
        outJson["buildinfo"]["commit_author"] = GitMetadata::AuthorName();
        outJson["buildinfo"]["commit_date"] = GitMetadata::CommitDate();
        outJson["results"] = {};

        for(size_t searchResult = 0; searchResult < fileRanking.size(); searchResult++) {
            outJson["results"].emplace_back();
            outJson["results"][searchResult] = {};
            outJson["results"][searchResult]["name"] = haystackFiles.at(fileRanking.at(searchResult).fileID).filename().string();
            outJson["results"][searchResult]["score"] = fileRanking.at(searchResult).score;
            outJson["results"][searchResult]["imageID"] = fileRanking.at(searchResult).imageID;
            outJson["results"][searchResult]["executionTimeSeconds"] = fileRanking.at(searchResult).executionTimeSeconds;
        }

        std::ofstream outFile(outputFile.value());
        outFile << outJson.dump(4);
        outFile.close();
    }

    // Free memory
    ShapeDescriptor::free::array(descriptorOrigins);
    ShapeDescriptor::free::array(descriptors);
    ShapeDescriptor::free::mesh(mesh);
    ShapeDescriptor::free::mesh(gpuMesh);
}