#include <iostream>
#include <arrrgh.hpp>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/fileutils.h>
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

int main(int argc, const char** argv) {
    const unsigned int numberOfSearchResultsToGenerate = 1;
    const float supportRadius = 100.0;


    arrrgh::parser parser("occurrenceCounter", "Count bits of descriptors that have been indexed.");
    const auto &indexDirectory = parser.add<std::string>(
            "index-directory", "The directory containing the index to be counted.", '\0',
            arrrgh::Required, "");
    const auto &outputFile = parser.add<std::string>(
            "output-file", "The file the results should be written to.", '\0',
            arrrgh::Required, "");

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

    std::cout << "Build info: " << GitMetadata::CommitSHA1() << ", by " << GitMetadata::AuthorName() << " on " << GitMetadata::CommitDate() << std::endl;

    std::cout << "Reading cluster file.." << std::endl;
    Cluster* cluster = readCluster(cluster::path(indexDirectory.value()) / "index.dat");
    std::cout << "\tCluster contains " << cluster->nodes.size() << " nodes and " << cluster->images.size() << " images." << std::endl;

    ShapeDescriptor::RICIDescriptor counts;
    std::fill(counts.contents, counts.contents + (spinImageWidthPixels * spinImageWidthPixels), 0);
    for(unsigned int i = 0; i < cluster->images.size(); i++) {
        for(unsigned int bit = 0; bit < spinImageWidthPixels * spinImageWidthPixels; bit++) {
            unsigned int chunkIndex = bit / 32;
            unsigned int bitIndex = bit % 32;

            unsigned int chunk = cluster->images.at(i).contents[chunkIndex];
            unsigned int bitValue = (chunk >> (31 - bitIndex)) & 0x1;
            counts.contents[bit] += bitValue;
        }
    }

    std::ofstream outFile(outputFile.value());
    for(int row = spinImageWidthPixels - 1; row >= 0; row--) {
        for(unsigned int col = 0; col < spinImageWidthPixels; col++) {
            outFile << counts.contents[spinImageWidthPixels * row + col] << (col == spinImageWidthPixels - 1 ? "" : ", ");
        }
        outFile << std::endl;
    }

    outFile.close();

    delete cluster;
}