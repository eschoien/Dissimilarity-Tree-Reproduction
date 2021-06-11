#pragma once

#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/gpu/types/array.h>

ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> convertRICIToModifiedQUICCI(ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors);