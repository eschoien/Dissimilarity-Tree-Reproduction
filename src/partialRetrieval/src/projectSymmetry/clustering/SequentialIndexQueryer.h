#pragma once

#include <projectSymmetry/utilities/customWeightedHamming.cuh>
#include "IndexQueryer.h"

namespace cluster {
    std::vector<QueryResult> sequentialQuery(std::experimental::filesystem::path dumpDirectory,
                                             const ShapeDescriptor::QUICCIDescriptor &queryImage,
                                             unsigned int resultCount,
                                             unsigned int fileStartIndex,
                                             unsigned int fileEndIndex,
                                             unsigned int num_threads = 0,
                                             debug::QueryRunInfo* runInfo = nullptr);
    void searchSingleFile(const ShapeDescriptor::QUICCIDescriptor &queryImage,
                          unsigned int resultCount,
                          const ProjectSymmetry::utilities::HammingWeights &hammingWeights,
                          std::set<cluster::QueryResult> &searchResults,
                          const ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> &images,
                          float currentScoreThreshold = 0,
                          unsigned int fileIndex = 0);
}