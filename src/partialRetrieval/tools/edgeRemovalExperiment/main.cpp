#include <arrrgh.hpp>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <random>
#include <experimental/filesystem>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <projectSymmetry/descriptors/binaryRICIConverter.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <bitset>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include <json.hpp>
#include <tsl/ordered_map.h>
#include <git.h>
#include "Histogram.h"


template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;


struct VertexScore {
    // Number of bits in the query descriptor which should be set to 0 but are set to 1
    unsigned int wrongQueryBitCount = 0;
    // The number of bits overlapping with the reference descriptor
    unsigned int overlappingQueryBitCount = 0;
    // Number of bits set to 1 in the reference descriptor
    unsigned int referenceDescriptorSetBitCount = 0;
};

struct DescriptorScore {
    VertexScore originalScore;
    VertexScore modifedScore;
};

VertexScore computeVertexScore(ShapeDescriptor::QUICCIDescriptor &queryDescriptor,
                                ShapeDescriptor::QUICCIDescriptor &referenceDescriptor) {
    VertexScore scores;

    for(unsigned int chunk = 0; chunk < UINTS_PER_QUICCI; chunk++) {
        scores.wrongQueryBitCount += std::bitset<32>(queryDescriptor.contents[chunk] & ~referenceDescriptor.contents[chunk]).count();
        scores.overlappingQueryBitCount += std::bitset<32>(queryDescriptor.contents[chunk] & referenceDescriptor.contents[chunk]).count();
        scores.referenceDescriptorSetBitCount += std::bitset<32>(referenceDescriptor.contents[chunk]).count();
    }

    return scores;
}

int main(int argc, const char** argv) {
    arrrgh::parser parser("edgeRemovalExperiment", "Experiment to quantify the effectiveness of unwanted bit removal.");
    const auto& showHelp = parser.add<bool>("help", "Show this help message.", 'h', arrrgh::Optional, false);
    const auto& queryDirectory = parser.add<std::string>("query-directory", "Directory containing 3D mesh files of the partial queries", '\0', arrrgh::Required, "NOT_SELECTED");
    const auto& haystackDirectory = parser.add<std::string>("reference-object-directory", "Directory containing 3D mesh files of the meshes from which the queries were extracted. NOTE: corresponding partial and complete objects MUST have equivalent names!", '\0', arrrgh::Required, "NOT_SELECTED");
    const auto& outputFile = parser.add<std::string>("output-file", "Location where to dump the produced QUICCI descriptors", '\0', arrrgh::Required, "NOT_SELECTED");
    const auto &forceGPU = parser.add<int>("force-gpu", "Index of the GPU device to use for search kernels.", '\0', arrrgh::Optional, -1);
    const auto& supportRadius = parser.add<float>("support-radius", "The support radius to use during descriptor generation", '\0', arrrgh::Optional, 1);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    std::vector<std::experimental::filesystem::path> queryFiles = ShapeDescriptor::utilities::listDirectory(queryDirectory.value());
    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(haystackDirectory.value());

    cudaDeviceProp gpuProperties = ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());

    json outJson;

    outJson["version"] = "v8";
    outJson["supportRadius"] = supportRadius.value();
    outJson["queryObjectFilesDirectory"] = queryDirectory.value();
    outJson["referenceObjectFilesDirectory"] = haystackDirectory.value();
    outJson["outputFile"] = outputFile.value();
    outJson["spinImageWidthPixels"] = spinImageWidthPixels;

    outJson["buildinfo"] = {};
    outJson["buildinfo"]["commit"] = GitMetadata::CommitSHA1();
    outJson["buildinfo"]["commit_author"] = GitMetadata::AuthorName();
    outJson["buildinfo"]["commit_date"] = GitMetadata::CommitDate();

    outJson["gpuInfo"] = {};
    outJson["gpuInfo"]["name"] = gpuProperties.name;
    outJson["gpuInfo"]["clockrate"] = gpuProperties.clockRate;
    outJson["gpuInfo"]["memoryCapacityInMB"] = gpuProperties.totalGlobalMem / (1024 * 1024);

    unsigned int excludedReductionsCount = 0;
    unsigned int excludedOverlapsCount = 0;
    unsigned int excludedMissingMatchCount = 0;
    unsigned int processedDescriptorCount = 0;
    Histogram wrongBitCountReductions(0, 2, 200);
    Histogram originalWrongBitCounts(0, 4096, 4096);
    Histogram modifiedWrongBitCounts(0, 4096, 4096);
    Histogram originalOverlapWithReference(0, 1, 100);
    Histogram modifiedOverlapWithReference(0, 1, 100);

    for(unsigned int i = 0; i < haystackFiles.size(); i++) {
        std::cout << "Processing " << (i + 1) << "/" << haystackFiles.size() << ": " << haystackFiles.at(i) << std::endl;
        assert(queryFiles.at(i).filename().string() == haystackFiles.at(i).filename().string());

        std::cout << "  Loading meshes: " << std::flush;
        ShapeDescriptor::cpu::Mesh queryMesh = ShapeDescriptor::utilities::loadMesh(queryFiles.at(i));
        std::cout << "query " << std::flush;
        processedDescriptorCount += queryMesh.vertexCount;
        ShapeDescriptor::cpu::Mesh haystackMesh = ShapeDescriptor::utilities::loadMesh(haystackFiles.at(i), true);
        std::cout << "reference" << std::endl;

        std::cout << "  Computing descriptors: " << std::flush;
        ShapeDescriptor::gpu::Mesh queryMeshGPU = ShapeDescriptor::copy::hostMeshToDevice(queryMesh);
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> queryOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(queryMeshGPU);

        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> originalQUICCIDescriptorsGPU = ShapeDescriptor::gpu::generateQUICCImages(queryMeshGPU, queryOrigins, supportRadius.value());
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> originalQUICCIDescriptors = ShapeDescriptor::copy::deviceArrayToHost(originalQUICCIDescriptorsGPU);
        ShapeDescriptor::free::array(originalQUICCIDescriptorsGPU);
        std::cout << "query_original " << std::flush;

        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> queryRICIDescriptors = ShapeDescriptor::gpu::generateRadialIntersectionCountImages(queryMeshGPU, queryOrigins, supportRadius.value());
        ShapeDescriptor::free::mesh(queryMeshGPU);
        ShapeDescriptor::free::array(queryOrigins);
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> modifiedQUICCIDescriptorsGPU = convertRICIToModifiedQUICCI(queryRICIDescriptors);
        ShapeDescriptor::free::array(queryRICIDescriptors);
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> modifiedQUICCIDescriptors = ShapeDescriptor::copy::deviceArrayToHost(modifiedQUICCIDescriptorsGPU);
        ShapeDescriptor::free::array(modifiedQUICCIDescriptorsGPU);
        std::cout << "query_modified " << std::flush;

        ShapeDescriptor::gpu::Mesh haystackMeshGPU = ShapeDescriptor::copy::hostMeshToDevice(haystackMesh);
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> haystackOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(haystackMeshGPU);
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> referenceQUICCIDescriptorsGPU = ShapeDescriptor::gpu::generateQUICCImages(haystackMeshGPU, haystackOrigins, supportRadius.value());
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> referenceQUICCIDescriptors = ShapeDescriptor::copy::deviceArrayToHost(referenceQUICCIDescriptorsGPU);
        ShapeDescriptor::free::array(haystackOrigins);
        ShapeDescriptor::free::mesh(haystackMeshGPU);
        ShapeDescriptor::free::array(referenceQUICCIDescriptorsGPU);
        std::cout << "reference" << std::endl;



        std::cout << "  Comparing descriptors.." << std::endl;

        // for each query vertex
        // - find matching reference vertex (must be identical)
        // - compare descriptors or reference and modified methods

        #pragma omp parallel for schedule(dynamic)
        for(unsigned int queryVertexIndex = 0; queryVertexIndex < queryMesh.vertexCount; queryVertexIndex++) {
            unsigned int matchingReferenceVertexIndex = 0xFFFFFFFF;
            ShapeDescriptor::cpu::float3 queryVertex = queryMesh.vertices[queryVertexIndex];
            ShapeDescriptor::cpu::float3 queryNormal = queryMesh.normals[queryVertexIndex];

            for(unsigned int referenceVertexIndex = 0; referenceVertexIndex < haystackMesh.vertexCount; referenceVertexIndex++) {
                ShapeDescriptor::cpu::float3 referenceVertex = haystackMesh.vertices[referenceVertexIndex];
                ShapeDescriptor::cpu::float3 referenceNormal = haystackMesh.normals[referenceVertexIndex];

                // Reading ascii files means that sometimes the exact floats slightly differ.
                // We therefore need to allow for a small error between otherwise equivalent vertices and normals.
                bool verticesEqual = length(queryVertex - referenceVertex) < 0.001;
                bool normalsEqual =
                        (std::abs(queryNormal.x - referenceNormal.x) < 0.0001) &&
                        (std::abs(queryNormal.y - referenceNormal.y) < 0.0001) &&
                        (std::abs(queryNormal.z - referenceNormal.z) < 0.0001);
                if(verticesEqual && normalsEqual) {
                    matchingReferenceVertexIndex = referenceVertexIndex;
                    break;
                }
            }

            if(matchingReferenceVertexIndex == 0xFFFFFFFF) {
                std::cout << "WARNING: vertex " << queryVertexIndex << " has no matching vertex in the reference mesh!\nThis vertex will be skipped and excluded from the results." << std::endl;
                #pragma omp atomic
                excludedMissingMatchCount++;
                continue;
            }

            DescriptorScore score;

            score.originalScore = computeVertexScore(
                    originalQUICCIDescriptors.content[queryVertexIndex],
                    referenceQUICCIDescriptors.content[matchingReferenceVertexIndex]);
            score.modifedScore = computeVertexScore(
                    modifiedQUICCIDescriptors.content[queryVertexIndex],
                    referenceQUICCIDescriptors.content[matchingReferenceVertexIndex]);
            //std::cout << originalVertexScore << " vs " << modifiedVertexScore << std::endl;

            /*if(modifiedVertexScore > originalVertexScore) {
                ShapeDescriptor::print::quicciDescriptor(originalQUICCIDescriptors.content[queryVertexIndex]);
                ShapeDescriptor::print::quicciDescriptor(modifiedQUICCIDescriptors.content[queryVertexIndex]);
                ShapeDescriptor::print::quicciDescriptor(referenceQUICCIDescriptors.content[matchingReferenceVertexIndex]);
            }*/

            if(score.originalScore.wrongQueryBitCount != 0) {
                wrongBitCountReductions.count(
                        double(score.modifedScore.wrongQueryBitCount) /
                        double(score.originalScore.wrongQueryBitCount));
            } else {
                #pragma omp atomic
                excludedReductionsCount++;
            }

            #pragma omp atomic
            originalWrongBitCounts.contents.at(score.originalScore.wrongQueryBitCount).count++;
            #pragma omp atomic
            modifiedWrongBitCounts.contents.at(score.modifedScore.wrongQueryBitCount).count++;

            if(score.originalScore.referenceDescriptorSetBitCount != 0) {
                originalOverlapWithReference.count(
                        double(score.originalScore.overlappingQueryBitCount) /
                        double(score.originalScore.referenceDescriptorSetBitCount));
                modifiedOverlapWithReference.count(
                        double(score.modifedScore.overlappingQueryBitCount) /
                        double(score.modifedScore.referenceDescriptorSetBitCount));
            } else {
                #pragma omp atomic
                excludedOverlapsCount++;
            }
        }

        ShapeDescriptor::free::array(modifiedQUICCIDescriptors);
        ShapeDescriptor::free::array(originalQUICCIDescriptors);
        ShapeDescriptor::free::array(referenceQUICCIDescriptors);
        ShapeDescriptor::free::mesh(queryMesh);
        ShapeDescriptor::free::mesh(haystackMesh);
    }

    std::cout << "Writing output file to " << outputFile.value() << std::endl;

    outJson["excludedReductionsCount"] = excludedReductionsCount;
    outJson["excludedOverlapsCount"] = excludedOverlapsCount;
    outJson["excludedMissingMatchCount"] = excludedMissingMatchCount;
    outJson["processedDescriptorCount"] = processedDescriptorCount;
    
    outJson["reductionInWrongBitCounts"] = {};
    for(unsigned int i = 0; i < wrongBitCountReductions.binCount; i++) {
        outJson["reductionInWrongBitCounts"].emplace_back();
        outJson["reductionInWrongBitCounts"][i] = {};
        outJson["reductionInWrongBitCounts"][i]["min"] = wrongBitCountReductions.contents.at(i).min;
        outJson["reductionInWrongBitCounts"][i]["max"] = wrongBitCountReductions.contents.at(i).max;
        outJson["reductionInWrongBitCounts"][i]["count"] = wrongBitCountReductions.contents.at(i).count;
    }
    for(unsigned int i = 0; i < originalWrongBitCounts.binCount; i++) {
        outJson["originalWrongBitCounts"].emplace_back();
        outJson["originalWrongBitCounts"][i] = {};
        outJson["originalWrongBitCounts"][i]["min"] = originalWrongBitCounts.contents.at(i).min;
        outJson["originalWrongBitCounts"][i]["max"] = originalWrongBitCounts.contents.at(i).max;
        outJson["originalWrongBitCounts"][i]["count"] = originalWrongBitCounts.contents.at(i).count;
    }
    for(unsigned int i = 0; i < modifiedWrongBitCounts.binCount; i++) {
        outJson["modifiedWrongBitCounts"].emplace_back();
        outJson["modifiedWrongBitCounts"][i] = {};
        outJson["modifiedWrongBitCounts"][i]["min"] = modifiedWrongBitCounts.contents.at(i).min;
        outJson["modifiedWrongBitCounts"][i]["max"] = modifiedWrongBitCounts.contents.at(i).max;
        outJson["modifiedWrongBitCounts"][i]["count"] = modifiedWrongBitCounts.contents.at(i).count;
    }
    for(unsigned int i = 0; i < originalOverlapWithReference.binCount; i++) {
        outJson["originalOverlapWithReference"].emplace_back();
        outJson["originalOverlapWithReference"][i] = {};
        outJson["originalOverlapWithReference"][i]["min"] = originalOverlapWithReference.contents.at(i).min;
        outJson["originalOverlapWithReference"][i]["max"] = originalOverlapWithReference.contents.at(i).max;
        outJson["originalOverlapWithReference"][i]["count"] = originalOverlapWithReference.contents.at(i).count;
    }
    for(unsigned int i = 0; i < modifiedOverlapWithReference.binCount; i++) {
        outJson["modifiedOverlapWithReference"].emplace_back();
        outJson["modifiedOverlapWithReference"][i] = {};
        outJson["modifiedOverlapWithReference"][i]["min"] = modifiedOverlapWithReference.contents.at(i).min;
        outJson["modifiedOverlapWithReference"][i]["max"] = modifiedOverlapWithReference.contents.at(i).max;
        outJson["modifiedOverlapWithReference"][i]["count"] = modifiedOverlapWithReference.contents.at(i).count;
    }

    std::ofstream outFile(outputFile.value());
    outFile << outJson.dump(4);
    outFile.close();

    std::cout << "Complete." << std::endl;
}