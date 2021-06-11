#include <iostream>
#include <arrrgh.hpp>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <projectSymmetry/descriptors/binaryRICIConverter.h>
#include <json.hpp>
#include <json/tsl/ordered_map.h>
#include <git.h>
#include <projectSymmetry/clustering/IndexQueryer.h>
#include <projectSymmetry/clustering/ClusterIO.h>
#include <shapeDescriptor/utilities/weightedHamming.cuh>
#include <random>
#include <shapeDescriptor/utilities/dump/descriptorImages.h>

std::vector<unsigned int> interestingNodeIDs;

const std::vector<unsigned int> interestingIDs = {
        4703570,	9461369,	2832463,	4615693,	9358571,
        4715117,	2027664,	8813780,	8662915,	3565697,
        3288396,	618088, 	1345241,	1978672,	863878,
        6570353,	5548528,	7188837,	9302898,	2174824,
        4648216,	2488372,	8701581,	9683957,	7088782,
        3230579,	8386516,	5452867,	542864, 	581075,
        1571136,	3016270,	2791028,	7909294,	7444858,
        5008289,	4667478,	1509458,	3335756,	3742248,
        547451, 	5524414,	750844,     2549517,	4884044,
        5497243,	4739495,	5757189,	4595311,	9751420,
        9057271,	9781386,	1819860,	5395535,	6670534,
        7925550,	2314172,	1410237,	2841507,	3129407,
        9677272,	1453196,	5167340,	6558957,	6832562,
        70776,  	9612385,	6493631,	6168417,	5270182,
        3279319,	9906511,	4495242,	3726708,	6868523,
        8209046,	8821796,	5944276,	5774050,	9298967,
        312402, 	7815689,	1279941,	2888992,	9909019,
        8849189,	1028018,	6025440,	4080968,	754777,
        3434078,	8984626,	6621246,	4834505,	4573843,
        6384160,	4772636,	143143, 	6783498,	4932346
};

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

float computeMinWeightedHammingDistance(ShapeDescriptor::utilities::HammingWeights hammingWeights,
                                        const ShapeDescriptor::QUICCIDescriptor &needle,
                                        const ShapeDescriptor::QUICCIDescriptor &andImage,
                                        const ShapeDescriptor::QUICCIDescriptor &orImage) {
    // We can determine each type of mismatch that _must_ occur based on the sum and product images
    // Type 1: Query has bits set, that will never be set in the specific branch
    unsigned int minMissedSetBitCount = 0;
    // Type 2: Query has bits unset, that will always be set in the specific branch
    unsigned int minMissedUnsetBitCount = 0;
    for(int i = 0; i < ShapeDescriptor::QUICCIDescriptorLength; i++) {
        // Query bits that are 1, but always 0 in the image set
        minMissedSetBitCount += std::bitset<32>(needle.contents[i] & (~orImage.contents[i])).count();
        // Query bits that are 0, but always 1 in the image set
        minMissedUnsetBitCount += std::bitset<32>((~needle.contents[i]) & andImage.contents[i]).count();
    }

    return (hammingWeights.missingSetBitPenalty * float(minMissedSetBitCount))
           + (hammingWeights.missingUnsetBitPenalty * float(minMissedUnsetBitCount));
}

void dumpNodeStackImages(unsigned int nodeID, std::vector<TreeNode> &extendedNodeStack) {
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> nodeStackImages(extendedNodeStack.size());

    for(int i = 0; i < extendedNodeStack.size(); i++) {
        nodeStackImages.content[i] = extendedNodeStack.at(i).sumImage;
    }

    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> nodeImage;
    nodeImage.length = 1;
    nodeImage.content = &extendedNodeStack.at(extendedNodeStack.size() - 1).sumImage;

    ShapeDescriptor::dump::descriptors(nodeStackImages, "cluster_node_" + std::to_string(nodeID) + "_stack.png", 50);
    ShapeDescriptor::dump::descriptors(nodeImage, "node_image_" + std::to_string(nodeID) + ".png", 1);
}

void walk(Cluster *cluster, unsigned int nodeID, std::vector<TreeNode> &nodeStack, std::vector<unsigned int> &nodeIDs, ShapeDescriptor::QUICCIDescriptor &queryImage) {
    TreeNode node = cluster->nodes.at(nodeID);
    bool isLeafNode = node.matchingNodeID == 0xFFFFFFFF;
    std::vector<TreeNode> extendedNodeStack = nodeStack;
    extendedNodeStack.push_back(node);
    std::vector<unsigned int> extendedNodeIDs = nodeIDs;
    extendedNodeIDs.push_back(nodeID);

    if(!isLeafNode) {
        if(std::find(interestingNodeIDs.begin(), interestingNodeIDs.end(), nodeID) != interestingNodeIDs.end()) {
            dumpNodeStackImages(nodeID, extendedNodeStack);
        }

        walk(cluster, node.matchingNodeID, extendedNodeStack, extendedNodeIDs, queryImage);
        walk(cluster, node.differingNodeID, extendedNodeStack, extendedNodeIDs, queryImage);
    } else {

        for(unsigned int imageID = node.subtreeStartIndex; imageID < node.subtreeEndIndex; imageID++) {
            if(std::find(interestingIDs.begin(), interestingIDs.end(), imageID) != interestingIDs.end()) {
                std::cout << "CONSIDERED!!" << std::endl;
                std::cout << nodeID << std::endl;
                ShapeDescriptor::utilities::HammingWeights queryWeights = ShapeDescriptor::utilities::computeWeightedHammingWeights(queryImage);
                std::cout << "Hamming weights: " << queryWeights.missingSetBitPenalty << ", " << queryWeights.missingUnsetBitPenalty << std::endl;

                ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> leafNodeContents;
                leafNodeContents.content = cluster->images.data() + node.subtreeStartIndex;
                leafNodeContents.length = node.subtreeEndIndex - node.subtreeStartIndex;
                ShapeDescriptor::dump::descriptors(leafNodeContents, "cluster_node_" + std::to_string(nodeID) + "_contents.png", 50);

                dumpNodeStackImages(nodeID, extendedNodeStack);

                std::cout << "QUERY IMAGE DISTANCE: " << ShapeDescriptor::utilities::computeWeightedHammingDistance(queryWeights, queryImage.contents, cluster->images.at(imageID).contents, spinImageWidthPixels, spinImageWidthPixels) << std::endl;
                ShapeDescriptor::print::quicciDescriptor(cluster->images.at(imageID));

            }
        }
    }
}

int main(int argc, const char** argv) {
    const float supportRadius = 100.0;


    arrrgh::parser parser("indexedSearchBenchmark", "Perform a sequential search through a list of descriptors.");
    const auto &indexDirectory = parser.add<std::string>(
            "index-directory", "The directory containing the index to be queried.", '\0',
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

    for(unsigned int i = 0; i < 128; i++) {
        interestingNodeIDs.push_back(i);
    }

    std::cout << "Build info: " << GitMetadata::CommitSHA1() << ", by " << GitMetadata::AuthorName() << " on " << GitMetadata::CommitDate() << std::endl;

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh("/mnt/NEXUS/datasets/SHREC2016_generated_queries_original/T175.obj", true);
    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(
            gpuMesh);

    // Compute the descriptor(s)
    //std::cout << "Computing descriptors.." << std::endl;
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> riciDescriptors =
            ShapeDescriptor::gpu::generateRadialIntersectionCountImages(gpuMesh, descriptorOrigins, supportRadius);

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = convertRICIToModifiedQUICCI(riciDescriptors);
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> queryDescriptors = ShapeDescriptor::copy::deviceArrayToHost(descriptors);

    cluster::debug::QueryRunInfo info;

    //std::cout << "Querying.." << std::endl;
    ShapeDescriptor::QUICCIDescriptor queryImage = queryDescriptors.content[46273];

    std::cout << "Query image:" << std::endl << std::endl;
    ShapeDescriptor::print::quicciDescriptor(queryImage);

    std::cout << "Reading cluster file.." << std::endl;
    Cluster* cluster = readCluster(cluster::path(indexDirectory.value()) / "index.dat");
    std::cout << "\tCluster contains " << cluster->nodes.size() << " nodes and " << cluster->images.size() << " images." << std::endl;

    std::vector<TreeNode> nodeStack;
    std::vector<unsigned int> nodeIDStack;
    walk(cluster, 0, nodeStack, nodeIDStack, queryImage);

    delete cluster;
}