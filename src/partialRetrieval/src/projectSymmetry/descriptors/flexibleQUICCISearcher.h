#pragma once
#include <shapeDescriptor/gpu/types/ImageSearchResults.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>

namespace ShapeDescriptor {
    namespace beta {
#define QUICCI_DISTANCE WEIGHTED_HAMMING_DISTANCE
//#define QUICCI_DISTANCE FLEXIBLE_CLUTTER_RESISTANT

#if QUICCI_DISTANCE == WEIGHTED_HAMMING_DISTANCE
        typedef float distanceType;
#else
        typedef unsigned int distanceType;
#endif

        ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<distanceType>> runFlexibleQUICCISearch(
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_haystackDescriptors,
                float* executionTimeSeconds = nullptr);
    }
}