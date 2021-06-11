#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/ImageSearchResults.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>

void visualise(ShapeDescriptor::cpu::Mesh mesh,
               ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<unsigned int>> searchResults,
               float scale, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors);