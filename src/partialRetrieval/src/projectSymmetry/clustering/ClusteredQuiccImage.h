#pragma once

#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <array>

class ClusteredQuiccImage {
public:
    static const unsigned int BLOCKS_PER_IMAGE = (spinImageWidthPixels * spinImageWidthPixels) / 16;
    std::array<unsigned short, BLOCKS_PER_IMAGE> content;

private:
    static std::array<unsigned short, BLOCKS_PER_IMAGE> blockify(ShapeDescriptor::QUICCIDescriptor &descriptor);

public:
    ClusteredQuiccImage() {}

    ClusteredQuiccImage(ShapeDescriptor::QUICCIDescriptor descriptor)
        : content(blockify(descriptor)) {}
};