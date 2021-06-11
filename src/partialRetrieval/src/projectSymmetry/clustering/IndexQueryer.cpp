#include "IndexQueryer.h"
#include <queue>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cassert>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <projectSymmetry/utilities/customWeightedHamming.cuh>

struct UnvisitedNode {
    UnvisitedNode(unsigned int unvisitedNodeID, float minDistance)
    : nodeID(unvisitedNodeID), minDistanceScore(minDistance) {}

    unsigned int nodeID;
    float minDistanceScore;

    // We want the open node priority queue to sort items by lowest score
    // Since the priority queue by default optimises for finding the highest sorted element,
    // we need to invert the sort order.
    bool operator< (const UnvisitedNode &right) const {
        return minDistanceScore > right.minDistanceScore;
    }
};

struct SearchResultEntry {
    SearchResultEntry(ImageEntryMetadata meta, float distanceScore, const ShapeDescriptor::QUICCIDescriptor &imageEntry)
        : imageMeta(meta), image(imageEntry), distanceScore(distanceScore) {}
    SearchResultEntry() : imageMeta(), image(), distanceScore(0) {}

    ImageEntryMetadata imageMeta;
    float distanceScore;
    ShapeDescriptor::QUICCIDescriptor image;

    bool operator< (const SearchResultEntry &right) const {
        return distanceScore < right.distanceScore;
    }
};

float computeMinWeightedHammingDistance(ProjectSymmetry::utilities::HammingWeights hammingWeights,
                                               const ShapeDescriptor::QUICCIDescriptor &needle,
                                               const ShapeDescriptor::QUICCIDescriptor &andImage,
                                               const ShapeDescriptor::QUICCIDescriptor &orImage) {
    // We can determine each type of mismatch that _must_ occur based on the sum and product images
    // Type 1: Query has bits set, that will never be set in the specific branch
    unsigned int minMissedSetBitCount = 0;
    // Type 2: Query has bits unset, that will always be set in the specific branch
   // unsigned int minMissedUnsetBitCount = 0;
    for(int i = 0; i < ShapeDescriptor::QUICCIDescriptorLength; i++) {
        // Query bits that are 1, but always 0 in the image set
        minMissedSetBitCount += std::bitset<32>(needle.contents[i] & (~orImage.contents[i])).count();
        // Query bits that are 0, but always 1 in the image set
      //  minMissedUnsetBitCount += std::bitset<32>((~needle.contents[i]) & andImage.contents[i]).count();
    }

    return (hammingWeights.missingSetBitPenalty * float(minMissedSetBitCount));
      //   + (hammingWeights.missingUnsetBitPenalty * float(minMissedUnsetBitCount));
}

inline float computeMinDistanceThreshold(std::vector<SearchResultEntry> &currentSearchResults) {
    return currentSearchResults.at(currentSearchResults.size() - 1).distanceScore;
}

inline void visitNode(
        const TreeNode &node,
        Cluster* cluster,
        const unsigned int nodeID,
        std::priority_queue<UnvisitedNode> &closedNodeQueue,
        std::vector<SearchResultEntry> &currentSearchResults,
        const ShapeDescriptor::QUICCIDescriptor &queryImage,
        const ProjectSymmetry::utilities::HammingWeights &queryWeights,
        unsigned int* scannedLeafNodeCount,
        float debug_nodeMinDistance) {



    const bool nodeIsIntermediateNode = node.matchingNodeID != 0xFFFFFFFF;

    float searchResultScoreThreshold = computeMinDistanceThreshold(currentSearchResults);
    //std::cout << "\rVisiting node " << debug_visitedNodeCount << " -> " << currentSearchResults.size() << " search results, " << closedNodeQueue.size() << " queued nodes, " << searchResultScoreThreshold  << " vs " << closedNodeQueue.top().minDistanceScore << " - " << nodeID << std::flush;

    if(nodeIsIntermediateNode) {
        const TreeNode matchingNode = cluster->nodes.at(node.matchingNodeID);
        const TreeNode differingNode = cluster->nodes.at(node.differingNodeID);

        float matchingNodeMinDistanceScore = computeMinWeightedHammingDistance(queryWeights, queryImage, matchingNode.productImage, matchingNode.sumImage);
        float differingNodeMinDistanceScore = computeMinWeightedHammingDistance(queryWeights, queryImage, differingNode.productImage, differingNode.sumImage);

        if(matchingNodeMinDistanceScore <= searchResultScoreThreshold) {
            closedNodeQueue.emplace(node.matchingNodeID, matchingNodeMinDistanceScore);
        }

        if(differingNodeMinDistanceScore <= searchResultScoreThreshold) {
            closedNodeQueue.emplace(node.differingNodeID, differingNodeMinDistanceScore);
        }
    } else {
        // Node is leaf node
        (*scannedLeafNodeCount)++;

        // Iterate over all images contained within it
        for(unsigned int imageID = node.subtreeStartIndex; imageID < node.subtreeEndIndex; imageID++) {
            float distanceScore = ProjectSymmetry::utilities::computeWeightedHammingDistance(
                    queryWeights, queryImage.contents, cluster->images[imageID].contents, spinImageWidthPixels, spinImageWidthPixels);

            // Only consider the image if it is potentially better than what's there already
            if(distanceScore <= searchResultScoreThreshold) {
                for(unsigned int i = currentSearchResults.size() - 1; i >= 0; i--) {
                    bool entryShouldBeInsertedHere = (i == 0) || (currentSearchResults.at(i - 1).distanceScore < distanceScore);

                    // Move entry forward, except the last one, which is effectively discarded
                    if(i < currentSearchResults.size() - 1) {
                        currentSearchResults.at(i + 1) = currentSearchResults.at(i);
                    }

                    if(entryShouldBeInsertedHere) {
                        SearchResultEntry entry(cluster->imageMetadata[imageID], distanceScore, cluster->images[imageID]);
                        currentSearchResults.at(i) = entry;
                        searchResultScoreThreshold = computeMinDistanceThreshold(currentSearchResults);
                        break;
                    }
                }
            }
        }
    }
}

std::vector<cluster::QueryResult> cluster::query(
        Cluster* cluster,
        const ShapeDescriptor::QUICCIDescriptor &queryImage,
        unsigned int resultCountLimit,
        debug::QueryRunInfo* runInfo) {

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::priority_queue<UnvisitedNode> closedNodeQueue;
    std::vector<SearchResultEntry> currentSearchResults;

    // Initialising search result list
    currentSearchResults.resize(resultCountLimit);
    ShapeDescriptor::QUICCIDescriptor emptyImage;
    std::fill(emptyImage.contents, emptyImage.contents + ShapeDescriptor::QUICCIDescriptorLength, 0);
    SearchResultEntry emptyEntry = {{0xFBAD, 0xBADBADFF}, std::numeric_limits<float>::max(), emptyImage};
    std::fill(currentSearchResults.begin(), currentSearchResults.end(), emptyEntry);

    closedNodeQueue.emplace(0, 0);
    unsigned int visitedNodeCount = 0;
    unsigned int scannedLeafNodeCount = 0;

    std::array<double, spinImageWidthPixels * spinImageWidthPixels> executionTimesSeconds;
    std::fill(executionTimesSeconds.begin(), executionTimesSeconds.end(), -1);

    //std::cout << "Query in progress.." << std::endl;
    std::chrono::steady_clock::time_point queryStartTime = std::chrono::steady_clock::now();

    ProjectSymmetry::utilities::HammingWeights queryWeights = ProjectSymmetry::utilities::computeWeightedHammingWeights(queryImage);

    // Iteratively add additional nodes until there's no chance any additional node can improve the best distance score
    while(  !closedNodeQueue.empty() &&
            computeMinDistanceThreshold(currentSearchResults) >= closedNodeQueue.top().minDistanceScore) {
        UnvisitedNode nextBestUnvisitedNode = closedNodeQueue.top();
        closedNodeQueue.pop();
        const TreeNode node = cluster->nodes.at(nextBestUnvisitedNode.nodeID);

        visitNode(node, cluster, nextBestUnvisitedNode.nodeID, closedNodeQueue, currentSearchResults, queryImage, queryWeights, &scannedLeafNodeCount, nextBestUnvisitedNode.minDistanceScore);

        visitedNodeCount++;

        // Chop off irrelevant search results
        if(currentSearchResults.size() > resultCountLimit) {
            currentSearchResults.erase(currentSearchResults.begin() + resultCountLimit, currentSearchResults.end());
        }

        // Time measurement
        //std::chrono::steady_clock::time_point queryEndTime = std::chrono::steady_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(queryEndTime - queryStartTime);
        //double timeUntilNow = double(duration.count()) / 1000000000.0;
        //executionTimesSeconds.at(closedNodeQueue.top().minDistanceScore) = timeUntilNow;

        //std::cout << "Search results: ";
        //for(int i = 0; i < currentSearchResults.size(); i++) {
        //    std::cout << currentSearchResults.at(i).distanceScore << ", ";
        //}
        //std::cout << std::endl;
        //std::cout << "Closed nodes: ";
        //for(int i = 0; i < debug_closedNodeQueue.size(); i++) {
        //    std::cout << debug_closedNodeQueue.at(i).minDistanceScore << "|" << debug_closedNodeQueue.at(i).nodeID << ", ";
        //}
        //std::cout << std::endl;
    }

    //std::cout << std::endl << "Query finished, " << computeMinDistanceThreshold(currentSearchResults) << " vs " << closedNodeQueue.top().minDistanceScore << std::endl;

    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    //std::cout << std::endl << "Query complete. " << std::endl;
    //std::cout << "\tTotal execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;
    //std::cout << "\tNumber of nodes visited: " << visitedNodeCount << "/" << cluster->nodes.size() << " of which " << scannedLeafNodeCount << " are leaf nodes." << std::endl;

    std::vector<cluster::QueryResult> queryResults;
    queryResults.reserve(currentSearchResults.size());

    for(int i = 0; i < currentSearchResults.size(); i++) {
        queryResults.push_back({currentSearchResults.at(i).imageMeta, (float)currentSearchResults.at(i).distanceScore, currentSearchResults.at(i).image});
    //    std::cout << "\tResult " << i << ": "
    //           "file " << currentSearchResults.at(i).imageMeta.fileID <<
    //           ", image " << currentSearchResults.at(i).imageMeta.imageID <<
    //            ", score " << currentSearchResults.at(i).distanceScore << std::endl;
    }

    if(runInfo != nullptr) {
        runInfo->totalQueryTime = double(duration.count()) / 1000.0;
        runInfo->threadCount = 1;
        runInfo->distanceTimes = executionTimesSeconds;
        runInfo->visitedNodeCount = visitedNodeCount;
        runInfo->scannedLeafNodeCount = scannedLeafNodeCount;
    }

    return queryResults;
}
