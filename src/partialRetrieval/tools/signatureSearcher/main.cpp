#include <arrrgh.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <atomic>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/read/QUICCIDescriptors.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <projectSymmetry/descriptors/quicciStats.h>
#include <projectSymmetry/descriptors/quicciStatsCPU.h>
#include "DescriptorSignature.h"





#include <string> 
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/read/QUICCIDescriptors.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <projectSymmetry/descriptors/quicciStats.h>
#include <projectSymmetry/descriptors/quicciStatsCPU.h>
#include "Signature.h"
#include <projectSymmetry/lsh/SignatureIO.h>
#include <projectSymmetry/lsh/SignatureBuilder.h>




double jaccardSimilarity(std::vector<int> needleSig, std::vector<int> candidateSig)
{

    /* test for jaccard similarity
    int main() {
        std::vector<int> v1{2,4,1};
        std::vector<int> v2{2,7,1};
        printf("%f\n", jaccardSimilarity(v1, v2));
        return 0;
    }
    */

    // --- May be removed
    if (needleSig.size() != candidateSig.size())
    {
        std::cout << "Error, signatures are of different sizes" << std::endl;
        return 0;
    }
    // ---

    // Check the implementation of Jaccard similarity, may not be correct
    // is currently the (order-sensitive??) intersection divided by the union
    // could also be the (order-sensitive??) intersection divided by the length

    unsigned int intersections = 0;

    for (unsigned int i = 0; i < needleSig.size(); i++)
    {
        if (needleSig[i] == candidateSig[i])
        {
            intersections++;
        }
    }

    double jaccard_index = intersections / ((double)needleSig.size() + (double)candidateSig.size() - intersections);

    return jaccard_index;
}







void runObjectQuery()
{

    // --- Measure execution time
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    // ---

    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(imageDumpDirectory);

    // loop through all objects
    for (unsigned int i = 0; i < haystackFiles.size(); i++)
    {

        // read object signatures struct
        // ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::read::QUICCIDescriptors(haystackFiles.at(i));

        // loop through all descriptor signatures for object

        // compute jaccard similarity

        // keep the closest match

        // --- Measure execution time
        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;
        // ---
    }
}

// copied from another file and somewhat modified, needs further changes
int main(int argc, const char **argv)
{
    arrrgh::parser parser("similaritysearcher", "Search for similar descriptors using LSH.");

    const auto &indexDirectory = parser.add<std::string>(
        "index-directory", "The directory containing the signatures to be queried.", '\0',
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
    // const auto &matchingObjectCount = parser.add<int>(
    //         "object-threshold", "Number of search result objects which surpass the consensus threshold to find.", '\0', arrrgh::Optional, 1);
    const auto &forceGPU = parser.add<int>(
        "force-gpu", "Index of the GPU device to use for search kernels.", '\0', arrrgh::Optional, -1);
    const auto &outputProgressionFile = parser.add<std::string>(jaccardSimilarity
                                                                "output-progression-file",
                                                                "Path to a csv file showing scores after every query.", '\0', arrrgh::Optional, "NONE_SELECTED");
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
    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(haystackDirectory.value());

    // Cluster* cluster = readCluster(cluster::path(indexDirectory.value()) / "index.dat");
    // import signatures someway as above comment#include <arrrgh.hpp>









/////// something weird happened here







// Function for creating n number of permuations of integers 0-1023.



int main(int argc, const char** argv) {
    arrrgh::parser parser("clusterbuilder", "Create indexes for QUICCI images.");
    const auto& indexDirectory = parser.add<std::string>(
            "index-directory", "The directory where the signature file should be stored.", '\0', arrrgh::Optional, "");
    const auto& sourceDirectory = parser.add<std::string>(
            "quicci-dump-directory", "The directory where binary dump files of QUICCI images are stored that should be indexed.", '\0', arrrgh::Optional, "");
    const auto& showHelp = parser.add<bool>(
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
    if(showHelp.value())
    {
        return 0;
    }

    std::cout << "Computing signatures from files in " << sourceDirectory.value() << "..." << std::endl;

    // Cluster* cluster = buildClusterFromDumpDirectory(sourceDirectory.value(), indexDirectory.value(), 32, forceCPU.value());
    // TODO: Generate the signatures ...

    unsigned int numberOfPermutations = 10;

    std::vector<DescriptorSignature> descriptorSignatures = buildSignaturesFromDumpDirectory(sourceDirectory.value(), indexDirectory.value(), numberOfPermutations);

    // print signatures

    // works but gives segfault for too many descriptors?
    // for (int i = 0; i < descriptorSignatures.size(); i++) {

    //     std::cout << descriptorSignatures[i].file_id << "-";
    //     std::cout << descriptorSignatures[i].descriptor_id << ": ";

    //     for (int j = 0; j < numberOfPermutations; j++) {
    //         std::cout << descriptorSignatures[i].signatures[j] << " ";
    //     }
    //     std::cout << " " << std::endl;
    // }

    std::cout << "Writing cluster file.." << std::endl;

    // writeCluster(cluster, cluster::path(indexDirectory.value()) / "index.dat");
    // TODO: Write signature file ... (not implemented)
    // written to file during buildSignatures

    std::cout << std::endl << "Done." << std::endl;
}

    std::vector<ObjectQueryResult> searchResults;

    unsigned int startIndex = subsetStartIndex.value();
    unsigned int endIndex = subsetEndIndex.value() != -1 ? subsetEndIndex.value() : queryFiles.size();
    for (unsigned int queryFile = startIndex; queryFile < endIndex; queryFile++)
    {
        std::cout << "Processing query " << (queryFile + 1) << "/" << endIndex << ": " << queryFiles.at(queryFile).string() << std::endl;

        ObjectQueryResult queryResult = runObjectQuery(
            queryFiles.at(queryFile), cluster, supportRadius.value(), seed.value(),
            resultsPerQuery.value(), consensusThreshold.value(), haystackFiles, outputProgressionFile.value(), progressionIterationLimit.value());

        searchResults.push_back(queryResult);

        /*
                if(outputFile.value() != "NONE_SELECTED") {
                    json outJson;

        std::cout << std::endl << "Done." << std::endl;
                    outJson["version"] = "v8";
                    outJson["resultCount"] = resultsPerQuery.value();
                    outJson["queryObjectSupportRadius"] = supportRadius.value();
                    outJson["buildinfo"] = {};
                    outJson["buildinfo"]["commit"] = GitMetadata::CommitSHA1();
                    outJson["buildinfo"]["commit_author"] = GitMetadata::AuthorName();
                    outJson["buildinfo"]["commit_date"] = GitMetadata::CommitDate();

                    outJson["cluster"] = {};
                    outJson["cluster"]["imageCount"] = c
        std::cout << std::endl << "Done." << std::endl;luster->images.size();
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
                            outJson["results"][resultInd
        std::cout << std::endl << "Done." << std::endl;ex]["searchResults"][i]["objectFilePath"] = haystackFintersect
                    outFile.close();
                }

                if(outputProgressionFile.value() != "NON
        std::cout << std::endl << "Done." << std::endl;E_SELECTED") {
                    // Only process one file if a progression file is generated.
                    break;
                }
            }

            delete cluster;
        }*/
    }

    std::cout << std::endl
              << "Done." << std::endl;
}