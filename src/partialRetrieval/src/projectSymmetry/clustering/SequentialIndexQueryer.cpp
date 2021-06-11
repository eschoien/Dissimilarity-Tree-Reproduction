#include <omp.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/read/QUICCIDescriptors.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <set>
#include <mutex>
#include <iostream>
#include <malloc.h>
#include "SequentialIndexQueryer.h"

void cluster::searchSingleFile(const ShapeDescriptor::QUICCIDescriptor &queryImage,
                      unsigned int resultCount,
                      const ProjectSymmetry::utilities::HammingWeights &hammingWeights,
                      std::set<cluster::QueryResult> &searchResults,
                      const ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> &images,
                      float currentScoreThreshold,
                      unsigned int fileIndex) {

    std::mutex searchResultLock;

    // For each image, register pixels in dump file
#pragma omp parallel for schedule(dynamic)
    for (unsigned int imageIndex = 0; imageIndex < images.length; imageIndex++) {
        ShapeDescriptor::QUICCIDescriptor combinedImage = images.content[imageIndex];

        float distanceScore = ProjectSymmetry::utilities::computeWeightedHammingDistance(hammingWeights, queryImage.contents, combinedImage.contents, spinImageWidthPixels, spinImageWidthPixels);
        if(distanceScore < currentScoreThreshold || searchResults.size() < resultCount) {
            searchResultLock.lock();
            ImageEntryMetadata entry = {(unsigned short) fileIndex, imageIndex};
            searchResults.insert({entry, distanceScore, combinedImage});
            if(searchResults.size() > resultCount) {
                // Remove worst search result
                searchResults.erase(std::prev(searchResults.end()));
                // Update score threshold
                currentScoreThreshold = std::prev(searchResults.end())->score;
            }
            searchResultLock.unlock();
        }
    }

    delete images.content;

    if(fileIndex % 1000 == 0) {
        malloc_trim(0);
    }
}

std::vector<cluster::QueryResult> sequentialQuery(std::experimental::filesystem::path dumpDirectory,
                                                  const ShapeDescriptor::QUICCIDescriptor &queryImage,
                                                  unsigned int resultCount,
                                                  unsigned int fileStartIndex,
                                                  unsigned int fileEndIndex,
                                                  unsigned int threadCount,
                                                  cluster::debug::QueryRunInfo* runInfo = nullptr)    {
    ProjectSymmetry::utilities::HammingWeights hammingWeights = ProjectSymmetry::utilities::computeWeightedHammingWeights(queryImage);

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::cout << "Listing files.." << std::endl;
    std::vector<std::experimental::filesystem::path> filesToIndex = ShapeDescriptor::utilities::listDirectory(dumpDirectory);
    std::cout << "\tFound " << filesToIndex.size() << " files." << std::endl;

    omp_set_nested(1);


    if(threadCount == 0) {
        #pragma omp parallel
        {
            threadCount = omp_get_num_threads();
        };
    }

    std::set<cluster::QueryResult> searchResults;
    float currentScoreThreshold = std::numeric_limits<float>::max();

    for (unsigned int fileIndex = fileStartIndex; fileIndex < fileEndIndex; fileIndex++) {
        // Reading image dump file
        std::experimental::filesystem::path archivePath = filesToIndex.at(fileIndex);
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> images = ShapeDescriptor::read::QUICCIDescriptors(archivePath);

        #pragma omp critical
        {
            searchSingleFile(queryImage, resultCount, hammingWeights, searchResults,
                             images, currentScoreThreshold, fileIndex);

            std::cout << "\rProcessing of file " << fileIndex + 1 << "/" << fileEndIndex << " complete. Current best score: " << currentScoreThreshold << "            " << std::flush;
        }
    }

    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl << "Query complete. " << std::endl;
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    std::vector<cluster::QueryResult> results(searchResults.begin(), searchResults.end());

    /*for(int i = 0; i < resultCount; i++) {
        std::cout << "Result " << i
                  << ": score " << results.at(i).score
                  << ", file " << results.at(i).entry.fileIndex
                  << ", image " << results.at(i).entry.imageIndex << std::endl;
    }*/

    if(runInfo != nullptr) {
        double queryTime = double(duration.count()) / 1000.0;
        runInfo->totalQueryTime = queryTime;
        runInfo->threadCount = threadCount;
        std::fill(runInfo->distanceTimes.begin(), runInfo->distanceTimes.end(), queryTime);
    }

    return results;
}