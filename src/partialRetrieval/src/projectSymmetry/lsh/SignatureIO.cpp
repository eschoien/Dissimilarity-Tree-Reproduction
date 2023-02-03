#include "SignatureIO.h"

void writeSignatures(ObjectSignature objectSig, const std::experimental::filesystem::path outputDirectory, const unsigned int number_of_permutations) {

    
    const std::experimental::filesystem::path outputFile = outputDirectory.string() + "T" + std::to_string(objectSig.file_id) + ".dat";
    // std::cout << outputFile << std::endl;
    std::ofstream outStream(outputFile.string(), std::ios::out | std::ios::binary);

    const char headerString[4] = "SIF";
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

        // unsigned long long numSignatures = descriptorSig.signatures.size();
        // outStream.write((const char*) &numSignatures, sizeof(unsigned int));

        outStream.write((const char*) descriptorSig.signatures.data(), number_of_permutations * sizeof(int));
        // for(int j = 0; j < numSignatures; j++) {
        //     unsigned int signature = descriptorSig.signatures[j];
        //     outStream.write((const char*) &signature, sizeof(unsigned int));
        // }
    }
}

ObjectSignature *readSignature(const std::experimental::filesystem::path indexFile, const unsigned int number_of_permutations) {
    std::ifstream inStream(indexFile, std::ios::in | std::ios::binary);

    ObjectSignature* objectSig = new ObjectSignature;

    char headerString[4];
    inStream.read(headerString, 4);

    assert(std::string(headerString) == "SIF");

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
    // std::cout << fileID << std::endl;

    unsigned long long descriptorCount;
    inStream.read((char*) &descriptorCount, sizeof(unsigned long long));
    // std::cout << descriptorCount << std::endl;

    objectSig->file_id = fileID;
    objectSig->descriptorSignatures.resize(descriptorCount);

    for(unsigned int i = 0; i < descriptorCount; i++) {
        unsigned int descriptorID;
        inStream.read((char*) &descriptorID, sizeof(unsigned int));
        // std::cout << descriptorID << std::endl;
        objectSig->descriptorSignatures[i].descriptor_id = descriptorID;

        // unsigned long long numSignatures;
        // inStream.read((char*) &numSignatures, sizeof(unsigned long long));
        // std::cout << numSignatures << std::endl;
        objectSig->descriptorSignatures[i].signatures.resize(number_of_permutations);

        inStream.read((char*) objectSig->descriptorSignatures[i].signatures.data(), number_of_permutations * sizeof(int));
        // for(int j = 0; j < numSignatures; j++) {
        //     unsigned int signature;
        //     inStream.read((char*) &signature, sizeof(unsigned int));
            
        // }
    }

    return objectSig;
}