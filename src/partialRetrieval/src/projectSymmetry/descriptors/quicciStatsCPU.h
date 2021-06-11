#pragma once

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <array>
#include <projectSymmetry/types/Cluster.h>

void computeOccurrenceCountsCPU(ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                             std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* outputCounts,
                             ShapeDescriptor::QUICCIDescriptor* ignoreMask);
void computeOccurrenceLevelsCPU(ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                             std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels>* bitOrder,
                             std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* outputLevels);
void rearrangeImagesByLevelCPU( ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
                             ShapeDescriptor::cpu::array<ImageEntryMetadata> metadataBatch,
                             std::array<unsigned short, spinImageWidthPixels * spinImageWidthPixels>* bitOrder,
                             std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels>* levels,
                             unsigned int pivotLevel);
void computeNodeMaskImagesCPU(Cluster* tree);