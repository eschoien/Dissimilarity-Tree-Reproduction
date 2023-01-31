#include "SignatureIO.h"

void writeSignatures(ObjectSignature objectSig, const std::experimental::filesystem::path outputDirectory) {

    
    const std::experimental::filesystem::path outputFile = outputDirectory.string() + "/T" + std::to_string(objectSig.file_id) + ".dat";
    
    std::ofstream outStream(outputFile.string(), std::ios::out | std::ios::binary);

    const char headerString[4] = "SIF";
    outStream.write(headerString, 4);

    std::string commitFingerprint = GitMetadata::CommitSHA1();
    outStream.write(commitFingerprint.c_str(), 40);

    for(int i = 0; i <= objectSig.descriptorSignatures.size(); i++) {
        DescriptorSignature descriptorSig = objectSig.descriptorSignatures[i];
        outStream.write((const char*) &descriptorSig.descriptor_id, sizeof(unsigned int));
        for(int j = 0; j <= descriptorSig.signatures.size(); j++) {
            outStream.write((const char*) &descriptorSig.signatures[j], sizeof(unsigned int));
        }
    }

    // unsigned long long imageWidth = spinImageWidthPixels;
    // outStream.write((const char*) &imageWidth, sizeof(unsigned long long));

    // unsigned long long nodeCount = cluster->nodes.size();
    // outStream.write((const char*) &nodeCount, sizeof(unsigned long long));

    // unsigned long long imageCount = cluster->images.size();
    // outStream.write((const char*) &imageCount, sizeof(unsigned long long));

    // unsigned long long indexedFileCount = cluster->indexedFiles.size();
    // outStream.write((const char*) &indexedFileCount, sizeof(unsigned long long));

    // outStream.write((const char*) &cluster->maxImagesPerLeafNode, sizeof(unsigned int));

    // outStream.write((const char*) cluster->nodes.data(), nodeCount * sizeof(TreeNode));

    // outStream.write((const char*) cluster->images.data(), imageCount * sizeof(ShapeDescriptor::QUICCIDescriptor));

    // outStream.write((const char*) cluster->imageMetadata.data(), imageCount * sizeof(ImageEntryMetadata));

    // for(cluster::path &filePath : cluster->indexedFiles) {
    //     std::string pathString = filePath.string();
    //     unsigned int pathLength = pathString.size();
    //     outStream.write((const char*) &pathLength, sizeof(unsigned int));
    //     outStream.write(pathString.data(), pathLength * sizeof(char));
    // }
}