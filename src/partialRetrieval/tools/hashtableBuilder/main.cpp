#include <vector>
#include <unordered_map>
#include <iostream>
#include <string>
#include <algorithm>
#include <projectSymmetry/lsh/Hashtable.h>
#include <projectSymmetry/lsh/HashtableIO.h>
#include <arrrgh.hpp>
#include <shapeDescriptor/utilities/fileutils.h>
#include <projectSymmetry/lsh/Signature.h>
#include <projectSymmetry/lsh/Permutation.h>
#include <random>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/read/QUICCIDescriptors.h>
#include <shapeDescriptor/utilities/free/array.h>

/* NOTES TO SELF:
    
    emplace_hint is useless for unordered map, useful for ordered

    maybe do this? once for each hashtable
    // ht.rehash(1024);

*/

/*
    bool operator==(const DescriptorEntry& lhs, const DescriptorEntry& rhs) {
        return lhs.objectID == rhs.objectID && lhs.descriptorID == rhs.descriptorID;
    }
*/


int highestHashtableValueSize(std::vector<Hashtable*> hashtables) {

    int max = 0;

    for (Hashtable* ht : hashtables) {
        for (int i = 0; i <= 1024; i++) {
            max = std::max(max, (int) ht->at(i).size());
        }
    }

    return max;
}

unsigned int objectIDfromPath(std::string path) {
    return std::stoi(path.substr(path.find("/T")+2));
}



int main(int argc, const char** argv) {
    arrrgh::parser parser("hashtableBuilder", "LSH hashtables builder for object QUICCI images.");
    const auto& indexDirectory = parser.add<std::string>(
        "index-directory", "The directory where the signature file should be stored.", '\0', arrrgh::Optional, "");
    const auto& sourceDirectory = parser.add<std::string>(
        "quicci-dump-directory", "The directory where binary dump files of QUICCI images are stored that should be indexed.", '\0', arrrgh::Optional, "");
    const auto& numberOfPermutations = parser.add<int>(
        "permutation-count", "The number of Minhash permutations / signature length", '\0', arrrgh::Optional, 10);
    const auto& descriptorsPerObjectLimit = parser.add<int>(
        "descriptorsPerObjectLimit", "descriptorsPerObjectLimit", '\0', arrrgh::Optional, 2000);
    const auto& seed = parser.add<int>(
        "randomSeed", "Random seed to use for determining the order of query images to visit.", '\0', arrrgh::Optional, 725948161);
    const auto& showHelp = parser.add<bool>(
        "help", "Show this help message.", 'h', arrrgh::Optional, false);

    try {
        parser.parse(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if (showHelp.value()) {
        return 0;
    }


    // PARAMETERS
    const std::experimental::filesystem::path &imageDumpDirectory = sourceDirectory.value();
    // const std::experimental::filesystem::path &outputDirectory = indexDirectory.value(); 
    int numPermutations = numberOfPermutations.value();
    uint descriptorLimit = descriptorsPerObjectLimit.value();
    size_t random_seed = seed.value();

    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(imageDumpDirectory);

    // RANDOM GENERATOR
    std::random_device rd("/dev/urandom");
    size_t randomSeed = seed != 0 ? seed : rd();
    std::minstd_rand0 generator{randomSeed};
    std::uniform_real_distribution<float> distribution(0, 1);


    // PERMUTATIONS
    std::vector<std::vector<int>> permutations = create_permutations(numPermutations, (size_t) 32895532);


    // RANDOM ORDER
    std::vector<unsigned int> order(350000); // (queryDescriptors.length);
    for (unsigned int s = 0; s < 350000; s++) {
        order.at(s) = s;
    }
    std::shuffle(order.begin(), order.end(), generator);

    // INITIALIZE H HASHTABLES
    std::vector<Hashtable*> hashtables(numPermutations);

    for (unsigned int h = 0; h < numPermutations; h++) {
        
        Hashtable* ht = new Hashtable;

        // STEP: initialize hashtable (maybe no need to yet)
        // 1024 keys with empty descriptor vector for each
        
        for (unsigned int i = 0; i <= 1024; i++) {
            ht->emplace(i, *(new HashtableValue));
        }

        hashtables[h] = ht;
    }

    std::cout << "Processing objects and descriptors.." << std::endl;

    for (unsigned int i = 0; i < haystackFiles.size(); i++) {

        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::read::QUICCIDescriptors(haystackFiles.at(i));

        unsigned int oID = objectIDfromPath(haystackFiles.at(i));

        std::cout << "Current object ID: " << std::to_string(oID) << std::endl;

        // loop through descriptors for current object
        for (unsigned int j = 0; j < descriptorsPerObjectLimit; j++) {

            unsigned int dID = order[j] % descriptors.length + 1;

            std::vector<int> signature;

            computeDescriptorSignature(descriptors.content[order[j] % descriptors.length], &signature, permutations);

            // loop through descriptor signature values
            for (unsigned int s = 0; s < signature.size(); s++) {
                hashtables[s]->at(signature[s]).push_back(*(new DescriptorEntry{oID, dID}));
            }
        }    

        // highest htv length in hashtables
        // std::cout << "Highest current length: " << highestHashtableValueSize(hashtables) << std::endl << std::endl;

        ShapeDescriptor::free::array(descriptors);
    }


    std::cout << "\nHashtable construction complete..\n" << std::endl;
    
    
    //TODO, MAYBE: verify that the descriptors are placed correctly here somehow

    for (unsigned int i = 0; i < numPermutations; i++) { //hashtables.size()
        std::cout << "Writing hashtable " << i << " to file.." << std::endl;
        writeHashtable(*hashtables[i], "output/lsh/hashtables/H" + std::to_string(i) + ".dat");
    }
    
    // delete ht;

    // ^NOTE: may be able to create all hashtables at once,
    //        or need to create them iteratively

    /*
    // --- REMOVE
    Hashtable* ht1 = readHashtable("output/lsh/hashtables/H1.dat");
    Hashtable* ht2 = readHashtable("output/lsh/hashtables/H2.dat");

    int ht1count = 0;
    int ht2count = 0;

    for(int i = 0; i <= 1024; i++) {
        ht1count += ht1->at(i).size();
        ht2count += ht2->at(i).size();
    }

    std::cout << "Hashtable 1 count:" << std::endl;
    std::cout << ht1count << std::endl;
    std::cout << "Hashtable 2 count:" << std::endl;
    std::cout << ht2count << std::endl;
    // ------------
    */
    
    
    return 0;
}