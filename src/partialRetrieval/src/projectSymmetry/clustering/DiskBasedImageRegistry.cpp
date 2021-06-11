//
// Created by bart on 15.12.2020.
//

#include <iostream>
#include <shapeDescriptor/utilities/compress/byteCompressor.h>
#include <fstream>
#include "DiskBasedImageRegistry.h"

void DiskBasedImageRegistry::close() {
    // Flush final chunk
    unsigned int absoluteID = nextImageID - 1;
    unsigned int chunkIndex = absoluteID / chunkSize;
    unsigned int entryCountInChunk = absoluteID % chunkSize;
    unsigned int chunkInBufferIndex = chunkIndex % bufferChunkCount;
    writeChunkFromBuffer(chunkIndex, chunkInBufferIndex, entryCountInChunk);

    // Writing chunk offsets
    size_t fileOffsetListLength = compressedChunkFileOffsets.size();
    registryFileHandle->write((const char*) &fileOffsetListLength, sizeof(size_t));
    registryFileHandle->write((const char*) compressedChunkFileOffsets.data(),
                              fileOffsetListLength * sizeof(ChunkInfo));

    unsigned long long offsetListStartPointer = headerSize + nextChunkStartPointer;
    registryFileHandle->seekg(headerPreambleSize);
    registryFileHandle->write((char*) &offsetListStartPointer, sizeof(size_t));

    registryFileHandle->close();

    // Cleaning up
    delete registryFileHandle;
    delete[] chunkAccumulationBuffer;
    delete[] chunkCompressionBuffer;
    delete[] completedChunkWriteCounts;
}

void DiskBasedImageRegistry::open(cluster::path registryFile, unsigned int chunkBufferSize) {
    std::cout << "Opening registry file at " << registryFile.string() << std::endl;
    create(registryFile, chunkBufferSize);

    // open registry

}

unsigned int DiskBasedImageRegistry::registerImage(ClusteredQuiccImage &image) {

    unsigned int reservedAbsoluteID = nextImageID++;

    unsigned int chunkIndex = reservedAbsoluteID / chunkSize;
    unsigned int chunkInBufferIndex = chunkIndex % bufferChunkCount;

    unsigned int reservedID = reservedAbsoluteID % chunkSize;

    if(reservedID == 0 && completedChunkWriteCounts[chunkInBufferIndex] == chunkSize) {
        throw std::runtime_error("Compression Buffer Overflow error!");
    }

    if(reservedID == 0) {
        fileOffsetLock.lock();
        compressedChunkFileOffsets.emplace_back(0, 0, 0);
        fileOffsetLock.unlock();
    }

    // Copy image into in-memory buffer
    chunkAccumulationBuffer[chunkInBufferIndex].at(reservedID) = image;

    unsigned int writesCompleted = completedChunkWriteCounts[chunkInBufferIndex]++;
    // The ++ operation on the atomic variable returns the unaltered value, we need the incremented one.
    writesCompleted++;

    // Chunk is full. We need to write it to disk and clear the current one.
    if (writesCompleted == chunkSize) {
        writeChunkFromBuffer(chunkIndex, chunkInBufferIndex, chunkSize);
    }

    return reservedID;
}

void DiskBasedImageRegistry::writeChunkFromBuffer(unsigned int chunkIndex, unsigned int chunkInBufferIndex, unsigned int entryCountInChunk) {

    std::string status = "";
    for(int i = 0; i < bufferChunkCount; i++) {
        status += (completedChunkWriteCounts[i] == chunkSize ? "|" : "-");
    }

    std::cout << status << std::endl; //"Compressing chunk " << chunkInBufferIndex << " -> " << chunkIndex << std::endl;
    size_t compressedChunkSize = ShapeDescriptor::utilities::compressBytes(
            chunkCompressionBuffer[chunkInBufferIndex].data(),
            chunkSize * sizeof(ClusteredQuiccImage),
            chunkAccumulationBuffer[chunkInBufferIndex].data(),
            entryCountInChunk * sizeof(ClusteredQuiccImage));

    status = "";
    for(int i = 0; i < bufferChunkCount; i++) {
        status += (completedChunkWriteCounts[i] == chunkSize ? "|" : "-");
    }

    std::cout << status << std::endl; //"Finished compressing chunk " << chunkInBufferIndex << " -> " << chunkIndex << ", size: " << (entryCountInChunk * sizeof(ClusteredQuiccImage)/(1024.0*1024.0)) << "MB -> " << (compressedChunkSize/(1024.0*1024.0)) << "MB" << std::endl;

    // Writing chunk to disk
// This lock also ensures only one thread attempts to write to the output file at a time
    fileOffsetLock.lock();
    unsigned long long chunkEndPointer = nextChunkStartPointer + compressedChunkSize;
    compressedChunkFileOffsets.at(chunkIndex) = {nextChunkStartPointer + headerSize, chunkEndPointer + headerSize, entryCountInChunk};
    nextChunkStartPointer += compressedChunkSize;
    registryFileHandle->write(chunkCompressionBuffer[chunkInBufferIndex].data(), compressedChunkSize);
    fileOffsetLock.unlock();

    // Reset progress counter to mark the buffer entry as clean
    completedChunkWriteCounts[chunkInBufferIndex] = 0;
}

void DiskBasedImageRegistry::fetchImage(unsigned int imageID, ClusteredQuiccImage *outputImage) {

}

void DiskBasedImageRegistry::create(cluster::path registryFilePath, unsigned int chunkBufferSize) {
    std::cout << "Opening registry file at " << registryFilePath.string() << std::endl;
    nextImageID = 0;
    registryFile = registryFilePath;

    chunkAccumulationBuffer = new std::array<ClusteredQuiccImage, chunkSize>[chunkBufferSize];
    chunkCompressionBuffer = new std::array<char, chunkSize * sizeof(ClusteredQuiccImage)>[chunkBufferSize];

    completedChunkWriteCounts = new std::atomic<unsigned int>[chunkBufferSize];
    std::fill(completedChunkWriteCounts, completedChunkWriteCounts + chunkBufferSize, 0);

    bufferChunkCount = chunkBufferSize;

    // Create empty registry file
    const char header[5] = "IMRF";
    unsigned long long headerStartPointer = 0;

    registryFileHandle = new std::fstream(registryFile.string(), std::ios::out | std::ios::binary);

    registryFileHandle->write(header, 5 * sizeof(char));
    registryFileHandle->write((char*) &headerStartPointer, sizeof(size_t));
}

DiskBasedImageRegistry::ChunkInfo::ChunkInfo(unsigned long long start, unsigned long long end, unsigned int length) : startByte(start), endByte(end), length(length) {}
