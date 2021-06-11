#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include "ClusteredQuiccImage.h"
#include "projectSymmetry/types/filesystem.h"

const unsigned int chunkSize = 65536;

class DiskBasedImageRegistry {

private:
    struct ChunkInfo {
        ChunkInfo(unsigned long long start, unsigned long long end, unsigned int length);

        unsigned long long startByte;
        unsigned long long endByte;
        unsigned int length;
    };

    cluster::path registryFile;

    std::fstream* registryFileHandle = nullptr;

    // Next image to be registered will receive this ID
    std::atomic<unsigned int> nextImageID;

    // Pointers to the start of each compressed chunk within the registry container file
    std::vector<ChunkInfo> compressedChunkFileOffsets;
    unsigned long long nextChunkStartPointer = 0;
    std::mutex fileOffsetLock;
    const size_t headerPreambleSize = 5;
    const size_t headerSize = headerPreambleSize + sizeof(unsigned long long);

    // Buffer which holds on to images as they are being registered. When full, the chunk can be compressed and written to disk.
    std::array<ClusteredQuiccImage, chunkSize>* chunkAccumulationBuffer;
    // In the process of compressing a chunk and writing it to disk, this buffer holds on to chunks that are being compressed, or are compressed and ready to be written to disk
    std::array<char, chunkSize * sizeof(ClusteredQuiccImage)>* chunkCompressionBuffer;
    // Number of images that have finished writing to a particular chunk. Used to determine whether all writes have been completed to that chunk
    std::atomic<unsigned int>* completedChunkWriteCounts;
    // Number of chunks contained in the above buffers
    unsigned int bufferChunkCount = 0;

    void create(cluster::path registryFile, unsigned int chunkBufferSize);
    void writeChunkFromBuffer(unsigned int chunkIndex, unsigned int chunkInBufferIndex, unsigned int entryCountInChunk);

public:
    unsigned int registerImage(ClusteredQuiccImage &image);
    void open(cluster::path registryFile, unsigned int chunkBufferSize);
    void close();
    void fetchImage(unsigned int imageID, ClusteredQuiccImage* outputImage);
};
