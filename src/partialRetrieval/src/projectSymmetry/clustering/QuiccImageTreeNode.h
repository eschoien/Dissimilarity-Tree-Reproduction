#pragma once

#include <array>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <projectSymmetry/types/BoolArray.h>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <projectSymmetry/settings.h>

struct TreeNode {

    unsigned int subtreeStartIndex = 0xFFFFFFFF;
    unsigned int subtreeEndIndex = 0xFFFFFFFF;

    unsigned int matchingNodeID = 0xFFFFFFFF;
    unsigned int differingNodeID = 0xFFFFFFFF;

    ShapeDescriptor::QUICCIDescriptor productImage;
    ShapeDescriptor::QUICCIDescriptor sumImage;
};