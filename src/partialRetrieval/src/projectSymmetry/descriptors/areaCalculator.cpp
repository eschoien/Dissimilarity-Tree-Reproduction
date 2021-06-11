#include "areaCalculator.h"

double computeMeshArea(ShapeDescriptor::cpu::Mesh &mesh) {
    double totalArea = 0;

    for(unsigned int vertexIndex = 0; vertexIndex < mesh.vertexCount; vertexIndex += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = mesh.vertices[vertexIndex + 0];
        ShapeDescriptor::cpu::float3 vertex1 = mesh.vertices[vertexIndex + 1];
        ShapeDescriptor::cpu::float3 vertex2 = mesh.vertices[vertexIndex + 2];

        ShapeDescriptor::cpu::float3 delta1 = vertex1 - vertex0;
        ShapeDescriptor::cpu::float3 delta2 = vertex2 - vertex0;

        double dx = delta1.y * delta2.z - delta2.y * delta1.z;
        double dy = delta1.x * delta2.z - delta2.x * delta1.z;
        double dz = delta1.x * delta2.y - delta2.x * delta1.y;

        double triangleArea = 0.5 * std::sqrt(dx * dx + dy * dy + dz * dz);

        totalArea += triangleArea;
    }

    return totalArea;
}
