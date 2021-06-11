#include <iostream>
#include <arrrgh.hpp>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <projectSymmetry/descriptors/areaCalculator.h>
#include <projectSymmetry/types/filesystem.h>
#include <json.hpp>
#include <json/tsl/ordered_map.h>
#include <git.h>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

int main(int argc, const char** argv) {
    arrrgh::parser parser("areaCalculator", "Compute the area of each 3D object in a directory.");
    const auto &objectDirectory = parser.add<std::string>(
            "object-directory", "The directory containing objects which should be processed.", '\0',
            arrrgh::Required, "");
    const auto &outputFile = parser.add<std::string>(
            "output-file", "Path to a CSV file to which to write the computed areas.", '\0', arrrgh::Required, "NONE_SELECTED");


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

    std::vector<cluster::path> directoryContents = ShapeDescriptor::utilities::listDirectory(objectDirectory.value());
    std::vector<double> areas(directoryContents.size());

    for(unsigned int i = 0; i < directoryContents.size(); i++) {
        std::cout << directoryContents.at(i) << std::endl;
        ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(directoryContents.at(i));
        areas.at(i) = computeMeshArea(mesh);
        ShapeDescriptor::free::mesh(mesh);
    }

    // libShapeDescriptor generators require a list of oriented points from which to compute descriptors
    // You can define these manually, or just grab all vertices from the input object
    // The latter is what the line below does.
    if(outputFile.value() != "NONE_SELECTED") {
        json outJson;

        outJson["version"] = "v6";
        outJson["buildinfo"] = {};
        outJson["buildinfo"]["commit"] = GitMetadata::CommitSHA1();
        outJson["buildinfo"]["commit_author"] = GitMetadata::AuthorName();
        outJson["buildinfo"]["commit_date"] = GitMetadata::CommitDate();
        outJson["areas"] = {};

        for(size_t searchResult = 0; searchResult < areas.size(); searchResult++) {
            outJson["areas"][directoryContents.at(searchResult).filename()] = areas.at(searchResult);
        }

        std::ofstream outFile(outputFile.value());
        outFile << outJson.dump(4);
        outFile.close();
    }
}