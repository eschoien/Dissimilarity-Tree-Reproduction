#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include "Hashtable.h"

std::string descriptorEntryToString(DescriptorEntry de) {
    std::string s;
    //s = "(O:" + std::to_string(de.objectID) + " D:" + std::to_string(de.descriptorID) + ")";
    s = std::to_string(de.objectID) + "-" + std::to_string(de.descriptorID);
    return s;
}


std::string hashtableValueToString(HashtableValue hv) {
    // maybe remove
    if (hv.size() == 0)
        return "[empty]";

    std::string s = "[";
    for (int i = 0; i < hv.size(); i++) {
        s += descriptorEntryToString(hv[i]); 
        i < hv.size()-1 ? s += ", " : s += "]";
    }
    return s;
}

void printStatistics(Hashtable* ht) {
    std::cout << "Bucket count: " << ht->bucket_count() << std::endl;
    std::cout << "Max bucket count: " << ht->max_bucket_count() << std::endl;
    std::cout << "Hashtable size: " << ht->size() << std::endl;
    std::cout << "Hashtable max size: " << ht->max_size() << std::endl;
    std::cout << "Load factor: " << ht->load_factor() << std::endl;
    std::cout << "Max load factor: " << ht->max_load_factor() << std::endl;
    
    //Used: bucket_count        : returns the number of buckets 
    //Used: max_bucket_count    : returns the maximum number of buckets
    //Used: size                : returns the number of elements
    //Used: max_size            : returns the maximum possible number of elements 
    // bucket_size              : returns the number of elements in specific bucket
    // bucket                   : returns the bucket for specific key
    //Used: load_factor         : returns average number of elements per bucket
    //Used: max_load_factor     : manages maximum average number of elements per bucket
}


void printHashtable(Hashtable* ht) {  
    for (const std::pair<const HashtableKey, HashtableValue>& n : *ht) {
        std::cout << n.first << " : " << hashtableValueToString(n.second) << std::endl;
    }
}

// maybe remove
// vector of hashtables
void printHashtables(std::vector<Hashtable*> hts) {
    //for (unsigned int i = 0; i < numPermutations; i++) { hashtables[i]
    for (Hashtable* ht : hts) {
        // std::cout << "\nHashtable: " << ht << std::endl;
        // std::cout << hashtableValueToString(ht->at(0)) << std::endl;
        std::cout << "\n" << ht << std::endl;
        // printHashtable(*ht);
        // printStatistics(ht);
    }
}

// Maybe restructure into namespace, possibly nested?