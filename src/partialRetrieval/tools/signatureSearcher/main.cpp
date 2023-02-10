#include <arrrgh.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <atomic>
#include <string> 
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

    //std::vector<std::experimental::filesystem::path> queryFiles = ShapeDescriptor::utilities::listDirectory(queryDirectory.value());
    //std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(haystackDirectory.value());

    // TODO:
    // load partial query objects

        // generate quicci descriptors
        
        // generate descriptor signatures

    // should have a objectSignature for the partial object at this point

        // loop through all descriptor signatures for object

        // compute jaccard similarity

        // keep the closest match




    /* COPIED FOR REFERENCE

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
    */


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

    std::cout << "Done." << std::endl;
}