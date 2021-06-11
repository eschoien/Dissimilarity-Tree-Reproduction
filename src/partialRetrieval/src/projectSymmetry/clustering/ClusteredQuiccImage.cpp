#include "ClusteredQuiccImage.h"

#if spinImageWidthPixels == 64

inline unsigned long reshuffleRow(const unsigned long row) {
    const unsigned long rowReshuffleMask0 = 0x000F000000000000;
    const unsigned long rowReshuffleMask1 = 0x00F0000F00000000;
    const unsigned long rowReshuffleMask2 = 0x0F0000F0000F0000;
    const unsigned long rowReshuffleMask3 = 0xF0000F0000F0000F;
    const unsigned long rowReshuffleMask4 = 0x0000F0000F0000F0;
    const unsigned long rowReshuffleMask5 = 0x00000000F0000F00;
    const unsigned long rowReshuffleMask6 = 0x000000000000F000;

    const unsigned long chunk0 = ((row << 36U) & rowReshuffleMask0);
    const unsigned long chunk1 = ((row << 24U) & rowReshuffleMask1);
    const unsigned long chunk2 = ((row << 12U) & rowReshuffleMask2);
    const unsigned long chunk3 = ((row       ) & rowReshuffleMask3);
    const unsigned long chunk4 = ((row >> 12U) & rowReshuffleMask4);
    const unsigned long chunk5 = ((row >> 24U) & rowReshuffleMask5);
    const unsigned long chunk6 = ((row >> 36U) & rowReshuffleMask6);

    return chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6;
}

inline long readRow(ShapeDescriptor::QUICCIDescriptor &descriptor, unsigned int row) {
    unsigned long chunk0 = descriptor.contents[2 * row + 0];
    unsigned long chunk1 = descriptor.contents[2 * row + 1];
    return (chunk0 << 32U) | chunk1;
}

inline void setRow(std::array<unsigned short, ClusteredQuiccImage::BLOCKS_PER_IMAGE> &descriptor, const unsigned int row, const unsigned long content) {
    descriptor[4 * row + 0] = (unsigned short) ((content >> 48U) & 0xFFFFU);
    descriptor[4 * row + 1] = (unsigned short) ((content >> 32U) & 0xFFFFU);
    descriptor[4 * row + 2] = (unsigned short) ((content >> 16U) & 0xFFFFU);
    descriptor[4 * row + 3] = (unsigned short) ((content >> 0U ) & 0xFFFFU);
}

std::array<unsigned short, ClusteredQuiccImage::BLOCKS_PER_IMAGE> ClusteredQuiccImage::blockify(ShapeDescriptor::QUICCIDescriptor &descriptor) {
    std::array<unsigned short, ClusteredQuiccImage::BLOCKS_PER_IMAGE> blockifiedDescriptor;

    for(unsigned int row = 0; row < spinImageWidthPixels; row += 4) {
        // Process the image in pairs of rows
        const unsigned long inputRow0 = readRow(descriptor, row + 0);
        const unsigned long inputRow1 = readRow(descriptor, row + 1);
        const unsigned long inputRow2 = readRow(descriptor, row + 2);
        const unsigned long inputRow3 = readRow(descriptor, row + 3);

        unsigned long long rowContents0 = reshuffleRow(inputRow0);
        unsigned long long rowContents1 = reshuffleRow(inputRow1);
        unsigned long long rowContents2 = reshuffleRow(inputRow2);
        unsigned long long rowContents3 = reshuffleRow(inputRow3);

        const unsigned long rowMask0 = 0xF000F000F000F000;
        const unsigned long rowMask1 = 0x0F000F000F000F00;
        const unsigned long rowMask2 = 0x00F000F000F000F0;
        const unsigned long rowMask3 = 0x000F000F000F000F;

        const unsigned long outRow0 = ((rowContents0 >> 0U) & rowMask0) |
                                      ((rowContents1 >> 4U) & rowMask1) |
                                      ((rowContents2 >> 8U) & rowMask2) |
                                      ((rowContents3 >> 12U) & rowMask3);

        const unsigned long outRow1 = ((rowContents0 << 4U) & rowMask0) |
                                      ((rowContents1 >> 0U) & rowMask1) |
                                      ((rowContents2 >> 4U) & rowMask2) |
                                      ((rowContents3 >> 8U) & rowMask3);

        const unsigned long outRow2 = ((rowContents0 << 8U) & rowMask0) |
                                      ((rowContents1 << 4U) & rowMask1) |
                                      ((rowContents2 >> 0U) & rowMask2) |
                                      ((rowContents3 >> 4U) & rowMask3);

        const unsigned long outRow3 = ((rowContents0 << 12U) & rowMask0) |
                                      ((rowContents1 << 8U) & rowMask1) |
                                      ((rowContents2 << 4U) & rowMask2) |
                                      ((rowContents3 >> 0U) & rowMask3);

        setRow(blockifiedDescriptor, row + 0, outRow0);
        setRow(blockifiedDescriptor, row + 1, outRow1);
        setRow(blockifiedDescriptor, row + 2, outRow2);
        setRow(blockifiedDescriptor, row + 3, outRow3);
    }

    return blockifiedDescriptor;
}
#else
std::array<unsigned short, ClusteredQuiccImage::BLOCKS_PER_IMAGE> ClusteredQuiccImage::blockify(ShapeDescriptor::QUICCIDescriptor &descriptor) {
    throw std::runtime_error("This function is not implemented for this image size!");
}
#endif