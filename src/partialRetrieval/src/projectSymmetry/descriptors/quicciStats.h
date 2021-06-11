#pragma once

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <array>
#include <projectSymmetry/types/Cluster.h>

void computeOccurrenceCounts(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                             std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* outputCounts,
                             ShapeDescriptor::QUICCIDescriptor* ignoreMask);
std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels> computePixelOrder(
                             std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* outputCounts);
void computeOccurrenceLevels(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                             std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels>* bitOrder,
                             std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* outputLevels);
unsigned int computePivotLevel(std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* levels,
                               unsigned int batchImageCount);
void rearrangeImagesByLevel( ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                             ShapeDescriptor::gpu::array<ImageEntryMetadata> metadataBatch,
                             std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels>* bitOrder,
                             std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* levels,
                             unsigned int pivotLevel);
void computeNodeMaskImages(  ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                             Cluster* tree);