#include <fstream>
#include <cassert>
#include <git.h>
#include <iostream>
#include "ClusterIO.h"

void writeCluster(Cluster *cluster, cluster::path outputFile) {
    std::ofstream outStream(outputFile.string(), std::ios::out | std::ios::binary);

    const char headerString[4] = "QIF";
    outStream.write(headerString, 4);

    std::string commitFingerprint = GitMetadata::CommitSHA1();
    outStream.write(commitFingerprint.c_str(), 40);

    unsigned long long imageWidth = spinImageWidthPixels;
    outStream.write((const char*) &imageWidth, sizeof(unsigned long long));

    unsigned long long nodeCount = cluster->nodes.size();
    outStream.write((const char*) &nodeCount, sizeof(unsigned long long));

    unsigned long long imageCount = cluster->images.size();
    outStream.write((const char*) &imageCount, sizeof(unsigned long long));

    unsigned long long indexedFileCount = cluster->indexedFiles.size();
    outStream.write((const char*) &indexedFileCount, sizeof(unsigned long long));

    outStream.write((const char*) &cluster->maxImagesPerLeafNode, sizeof(unsigned int));

    outStream.write((const char*) cluster->nodes.data(), nodeCount * sizeof(TreeNode));

    outStream.write((const char*) cluster->images.data(), imageCount * sizeof(ShapeDescriptor::QUICCIDescriptor));

    outStream.write((const char*) cluster->imageMetadata.data(), imageCount * sizeof(ImageEntryMetadata));

    for(cluster::path &filePath : cluster->indexedFiles) {
        std::string pathString = filePath.string();
        unsigned int pathLength = pathString.size();
        outStream.write((const char*) &pathLength, sizeof(unsigned int));
        outStream.write(pathString.data(), pathLength * sizeof(char));
    }
}

Cluster *readCluster(cluster::path indexFile) {
    std::ifstream inStream(indexFile, std::ios::in | std::ios::binary);

    Cluster* cluster = new Cluster;

    char headerString[4];
    inStream.read(headerString, 4);

    assert(std::string(headerString) == "QIF");

    char commitFingerprint[40];
    inStream.read(commitFingerprint, 40);
    cluster->indexFileCreationCommitHash = std::string(commitFingerprint);

    unsigned long long imageWidth;
    inStream.read((char*) &imageWidth, sizeof(unsigned long long));
    if(imageWidth != spinImageWidthPixels) {
        throw std::runtime_error("You're attempting to read an index of images with size " + std::to_string(imageWidth) + "x" + std::to_string(imageWidth) + ". "
                                      "The library is currently compiled to work for images of size " + std::to_string(spinImageWidthPixels) + "x" + std::to_string(spinImageWidthPixels) + ". "
                                      "These must match in order to work properly. Please recompile the library by editing the library settings file in libShapeDescriptor.");
    }

    unsigned long long nodeCount;
    inStream.read((char*) &nodeCount, sizeof(unsigned long long));

    unsigned long long imageCount;
    inStream.read((char*) &imageCount, sizeof(unsigned long long));

    unsigned long long indexedFileCount;
    inStream.read((char*) &indexedFileCount, sizeof(unsigned long long));

    unsigned int imagesPerBucket;
    inStream.read((char*) &imagesPerBucket, sizeof(unsigned int));

    cluster->nodes.resize(nodeCount);
    cluster->images.resize(imageCount);
    cluster->imageMetadata.resize(imageCount);
    cluster->indexedFiles.resize(indexedFileCount);
    cluster->maxImagesPerLeafNode = imagesPerBucket;

    inStream.read((char*) cluster->nodes.data(), nodeCount * sizeof(TreeNode));

    inStream.read((char*) cluster->images.data(), imageCount * sizeof(ShapeDescriptor::QUICCIDescriptor));

    inStream.read((char*) cluster->imageMetadata.data(), imageCount * sizeof(ImageEntryMetadata));

    for(unsigned int i = 0; i < indexedFileCount; i++) {
        unsigned int pathLength;
        inStream.read((char*) &pathLength, sizeof(unsigned int));
        std::string pathString;
        pathString.resize(pathLength);
        inStream.read((char*) pathString.data(), pathLength * sizeof(char));
        cluster->indexedFiles.at(i) = cluster::path(pathString);
    }

    return cluster;
}
