#pragma once

#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <projectSymmetry/types/Cluster.h>
#include <projectSymmetry/types/ClusterEntry.h>

namespace cluster {
    struct QueryResult {
        ImageEntryMetadata entry;
        float score = 0;
        ShapeDescriptor::QUICCIDescriptor image;

        bool operator<(const QueryResult &rhs) const {
            if (score != rhs.score) {
                return score < rhs.score;
            }

            return entry.imageID < rhs.entry.imageID;
        }
    };

    namespace debug {
        struct QueryRunInfo {
            double totalQueryTime = -1;
            unsigned int threadCount = 0;
            size_t visitedNodeCount = 0;
            size_t scannedLeafNodeCount = 0;
            std::array<double, spinImageWidthPixels * spinImageWidthPixels> distanceTimes;

            QueryRunInfo() {
                std::fill(distanceTimes.begin(), distanceTimes.end(), -1);
            }
        };
    }

    std::vector<QueryResult> query(Cluster *cluster, const ShapeDescriptor::QUICCIDescriptor &queryImage,
                                   unsigned int resultCountLimit, debug::QueryRunInfo *runInfo = nullptr);
}



