#pragma once

#include <vector>
#include <string>
#include <unordered_map>

struct DescriptorEntry {
    unsigned int objectID;
    unsigned int descriptorID;
};

// Key
typedef int HashtableKey;
// Value
typedef std::vector<DescriptorEntry> HashtableValue;
// Hashtable
typedef std::unordered_map<HashtableKey, HashtableValue> Hashtable;

std::string descriptorEntryToString(DescriptorEntry de);
std::string hashtableValueToString(HashtableValue hv);
void printStatistics(Hashtable* ht);
void printHashtable(Hashtable* ht);
void printHashtables(std::vector<Hashtable*> hts);