#pragma once

#include <fstream>
#include <cassert>
#include <git.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <experimental/filesystem>
#include "Hashtable.h"

void writeHashtable(const Hashtable& ht, const std::experimental::filesystem::path outputPath);
Hashtable* readHashtable(const std::experimental::filesystem::path inputPath);

/*
namespace lsh {
    namespace print {
        inline void hashtable(std::unordered_map<int, std::string> uo_map) {
            for (const auto& [key, value] : uo_map) {
                std::cout << key << " : " << value << std::endl;
            }
        }
    }
}
*/