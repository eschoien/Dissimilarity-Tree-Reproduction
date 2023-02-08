#include "SignatureIO.h"

void writeSignatures(ObjectSignature objectSig, const std::experimental::filesystem::path outputDirectory, const unsigned int numberOfPermutations) {

    
    const std::experimental::filesystem::path outputFile = outputDirectory.string() + "T" + std::to_string(objectSig.file_id) + ".dat";
    std::cout << outputFile << std::endl;
    std::ofstream outStream(outputFile.string(), std::ios::out | std::ios::binary);

    const char headerString[4] = "OSF";
    outStream.write(headerString, 4);

    // std::string commitFingerprint = GitMetadata::CommitSHA1();
    // outStream.write(commitFingerprint.c_str(), 40);

    unsigned long long fileID = objectSig.file_id;
    outStream.write((const char*) &fileID, sizeof(unsigned long long));

    unsigned long long descriptorCount = objectSig.descriptorSignatures.size();
    outStream.write((const char*) &descriptorCount, sizeof(unsigned long long));

    for(int i = 0; i < objectSig.descriptorSignatures.size(); i++) {
        DescriptorSignature descriptorSig;
        descriptorSig = objectSig.descriptorSignatures[i];

        unsigned int descriptorID = descriptorSig.descriptor_id;
        outStream.write((const char*) &descriptorID, sizeof(unsigned int));

        outStream.write((const char*) descriptorSig.signatures.data(), numberOfPermutations * sizeof(int));
    }
}

ObjectSignature *readSignature(const std::experimental::filesystem::path indexFile, const unsigned int numberOfPermutations) {
    std::ifstream inStream(indexFile, std::ios::in | std::ios::binary);

    ObjectSignature* objectSig = new ObjectSignature;

    char headerString[4];
    inStream.read(headerString, 4);

    assert(std::string(headerString) == "OSF");

    // char commitFingerprint[40];
    // inStream.read(commitFingerprint, 40);
    // cluster->indexFileCreationCommitHash = std::string(commitFingerprint);

    // unsigned long long imageWidth;
    // inStream.read((char*) &imageWidth, sizeof(unsigned long long));
    // if(imageWidth != spinImageWidthPixels) {
    //     throw std::runtime_error("You're attempting to read an index of images with size " + std::to_string(imageWidth) + "x" + std::to_string(imageWidth) + ". "
    //                                   "The library is currently compiled to work for images of size " + std::to_string(spinImageWidthPixels) + "x" + std::to_string(spinImageWidthPixels) + ". "
    //                                   "These must match in order to work properly. Please recompile the library by editing the library settings file in libShapeDescriptor.");
    // }

    unsigned long long fileID;
    inStream.read((char*) &fileID, sizeof(unsigned long long));

    unsigned long long descriptorCount;
    inStream.read((char*) &descriptorCount, sizeof(unsigned long long));

    objectSig->file_id = fileID;
    objectSig->descriptorSignatures.resize(descriptorCount);

    for(unsigned int i = 0; i < descriptorCount; i++) {
        unsigned int descriptorID;
        inStream.read((char*) &descriptorID, sizeof(unsigned int));
        objectSig->descriptorSignatures[i].descriptor_id = descriptorID;

        objectSig->descriptorSignatures[i].signatures.resize(numberOfPermutations);

        inStream.read((char*) objectSig->descriptorSignatures[i].signatures.data(), numberOfPermutations * sizeof(int));
    }

    return objectSig;
}

void writeSignatureIndex(SignatureIndex sigIndex, const std::experimental::filesystem::path outputFile) {

    std::ofstream outStream(outputFile.string(), std::ios::out | std::ios::binary);

    const char headerString[4] = "SIF";
    outStream.write(headerString, 4);

    unsigned long long fileCount = sigIndex.objectCount;
    outStream.write((const char*) &fileCount, sizeof(unsigned long long));

    unsigned long long numPermutations = sigIndex.numPermutations;
    outStream.write((const char*) &numPermutations, sizeof(unsigned long long));

    for (int i = 0; i < numPermutations; i++) {
        unsigned int permutation_id = i + 1;
        outStream.write((const char*) &permutation_id, sizeof(unsigned int));

        unsigned long long permutationCount = sigIndex.permutations[i].size();
        outStream.write((const char*) &permutationCount, sizeof(unsigned long long));

        outStream.write((const char*) sigIndex.permutations[i].data(), permutationCount * sizeof(int));
    }
}

SignatureIndex *readSignatureIndex(const std::experimental::filesystem::path indexFile) {
    std::ifstream inStream(indexFile, std::ios::in | std::ios::binary);

    SignatureIndex* sigIndex = new SignatureIndex;

    char headerString[4];
    inStream.read(headerString, 4);

    assert(std::string(headerString) == "SIF");

    unsigned long long fileCount;
    inStream.read((char*) &fileCount, sizeof(unsigned long long));
    sigIndex->objectCount = fileCount;

    unsigned long long numPermutations;
    inStream.read((char*) &numPermutations, sizeof(unsigned long long));

    sigIndex->numPermutations = numPermutations;
    sigIndex->permutations.resize(numPermutations);

    for (int i = 0; i < numPermutations; i++) {
        unsigned int permutation_id;
        inStream.read((char*) &permutation_id, sizeof(unsigned int));

        unsigned long long permutationCount;
        inStream.read((char*) &permutationCount, sizeof(unsigned long long));
        sigIndex->permutations[i].resize(permutationCount);

        inStream.read((char*) sigIndex->permutations[i].data(), permutationCount * sizeof(int));
    }

    return sigIndex;
}