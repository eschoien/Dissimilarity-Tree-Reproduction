#pragma once

#include "filesystem.h"
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <projectSymmetry/clustering/QuiccImageTreeNode.h>

struct ImageEntryMetadata {
    unsigned short fileID;
    unsigned int imageID;
};

struct Cluster {
    std::vector<ShapeDescriptor::QUICCIDescriptor> images;
    std::vector<ImageEntryMetadata> imageMetadata;
    std::vector<TreeNode> nodes;
    std::vector<cluster::path> indexedFiles;
    unsigned int maxImagesPerLeafNode;
    std::string indexFileCreationCommitHash;
};