#include "TestBlockification.h"

#include <catch2/catch.hpp>
#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <projectSymmetry/clustering/ClusteredQuiccImage.h>
#include <bitset>
#include <iostream>

TEST_CASE("Blockification of QUICCI descriptors") {


    SECTION("Blockfication process is a 1:1 bit mapping") {
        for(unsigned int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
            ShapeDescriptor::QUICCIDescriptor descriptor;
            std::fill(descriptor.contents, descriptor.contents + UINTS_PER_QUICCI, 0);

            unsigned int chunk = 0x80000000 >> i % 32;
            unsigned int chunkIndex = i / 32;
            descriptor.contents[chunkIndex] = chunk;

            ClusteredQuiccImage blockifiedImage(descriptor);

            unsigned int totalSetBitCount = 0;
            for (unsigned int blockIndex = 0; blockIndex < ClusteredQuiccImage::BLOCKS_PER_IMAGE; blockIndex++) {
                totalSetBitCount += std::bitset<16>(blockifiedImage.content[blockIndex]).count();
            }
            REQUIRE(totalSetBitCount == 1);
        }
    }

    SECTION("Blockfication maps to the correct block") {
        for(unsigned int blockX = 0; blockX < spinImageWidthPixels / 4; blockX++) {
            for(unsigned int blockY = 0; blockY < spinImageWidthPixels / 4; blockY++) {
                for(unsigned int bitX = 0; bitX < 4; bitX++) {
                    for(unsigned int bitY = 0; bitY < 4; bitY++) {

                        ShapeDescriptor::QUICCIDescriptor descriptor;
                        std::fill(descriptor.contents, descriptor.contents + UINTS_PER_QUICCI, 0);

                        unsigned int chunk = 0x80000000 >> ((4 * blockX + bitX) % 32);
                        unsigned int chunkIndex =
                                  2 * 4 * blockY
                                + 2 * bitY
                                + ((4 * blockX) / 32);

                        descriptor.contents[chunkIndex] = chunk;

                        ClusteredQuiccImage blockifiedImage(descriptor);

                        unsigned targetBlockIndex = (spinImageWidthPixels / 4) * blockY + blockX;
                        unsigned int totalSetBitCount = 0;
                        for (unsigned int blockIndex = 0; blockIndex < ClusteredQuiccImage::BLOCKS_PER_IMAGE; blockIndex++) {
                            unsigned int chunkSetBitCount = std::bitset<16>(blockifiedImage.content[blockIndex]).count();
                            if(blockIndex == targetBlockIndex) {
                                // Ensure bit ended up in the right chunk
                                REQUIRE(chunkSetBitCount == 1);
                                continue;
                            }

                            totalSetBitCount += chunkSetBitCount;
                        }

                        // Ensure no other bits are set
                        REQUIRE(totalSetBitCount == 0);
                    }
                }
            }
        }
    }

    SECTION("Blockfication process maps to unique bits") {
        ClusteredQuiccImage accumulatorImage;
        std::fill(accumulatorImage.content.begin(), accumulatorImage.content.end(), 0);

        for(unsigned int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
            ShapeDescriptor::QUICCIDescriptor descriptor;
            std::fill(descriptor.contents, descriptor.contents + UINTS_PER_QUICCI, 0);

            unsigned int chunk = 0x80000000 >> i % 32;
            unsigned int chunkIndex = i / 32;
            descriptor.contents[chunkIndex] = chunk;

            ClusteredQuiccImage blockifiedImage(descriptor);

            for (unsigned int blockIndex = 0; blockIndex < ClusteredQuiccImage::BLOCKS_PER_IMAGE; blockIndex++) {
                accumulatorImage.content[blockIndex] |= blockifiedImage.content[blockIndex];
            }
        }

        for (unsigned int blockIndex = 0; blockIndex < ClusteredQuiccImage::BLOCKS_PER_IMAGE; blockIndex++) {
            REQUIRE(accumulatorImage.content[blockIndex] == 0xFFFF);
        }
    }
}