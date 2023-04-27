#include <iostream>
#include <fstream>
#include <unordered_map>
//#include <boost/archive/binary_iarchive.hpp>
//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>
//#include <boost/archive/text_oarchive.hpp>
#include <string>
#include <experimental/filesystem>
#include "HashtableIO.h"

void writeHashtable(const Hashtable& ht, const std::experimental::filesystem::path outputPath) {

    //const std::experimental::filesystem::path outputFile = outputDirectory.string() + "H" + std::to_string(htID) + ".dat";

    std::cout << outputPath.string() << std::endl;
    
    std::ofstream outStream(outputPath.string(), std::ios::out | std::ios::binary);

    const char headerString[4] = "OSF";
    outStream.write(headerString, 4);

    for (int i = 0; i <= 1024; i++) {

        HashtableKey key = i;
        outStream.write((const char*) &key, sizeof(unsigned int));

        unsigned int htvLength = ht.at(i).size();
        outStream.write((const char*) &htvLength, sizeof(unsigned int));

        for (int d = 0; d < ht.at(i).size(); d++) {

            unsigned int oID = ht.at(i)[d].objectID;
            unsigned int dID = ht.at(i)[d].descriptorID;

            outStream.write((const char*) &oID, sizeof(unsigned int));
            outStream.write((const char*) &dID, sizeof(unsigned int));
        }
    }
}

Hashtable* readHashtable(const std::experimental::filesystem::path inputPath) {

    std::ifstream inStream(inputPath, std::ios::in | std::ios::binary);

    char headerString[4];
    inStream.read(headerString, 4);
    assert(std::string(headerString) == "OSF");

    Hashtable* ht = new Hashtable;

    for (int i = 0; i <= 1024; i++) {

        HashtableKey key;
        inStream.read((char*) &key, sizeof(unsigned int));

        unsigned int htvLength;
        inStream.read((char*) &htvLength, sizeof(unsigned int));

        HashtableValue* htv = new HashtableValue;
        htv->resize(htvLength);
        
        for (int d = 0; d < htvLength; d++) {

            unsigned int oID;
            inStream.read((char*) &oID, sizeof(unsigned int));

            unsigned int dID;
            inStream.read((char*) &dID, sizeof(unsigned int));

            DescriptorEntry de = {oID, dID};

            (*htv)[d] = de;
        }

        ht->emplace(key, *htv);
    }

    return ht;
}