#pragma once

#include "Signature.h"
#include <fstream>
#include <cassert>
#include <git.h>
#include <iostream>
#include <arrrgh.hpp>
#include <vector>
#include <atomic>
#include <string> 
#include <experimental/filesystem>

void writeSignatures(ObjectSignature objectSig, const std::experimental::filesystem::path outputDirectory, const unsigned int number_of_permutations);
ObjectSignature *readSignature(const std::experimental::filesystem::path indexFile, const unsigned int number_of_permutations);

void writeSignatureIndex(SignatureIndex sigIndex, const std::experimental::filesystem::path outputFile);
SignatureIndex *readSignatureIndex(const std::experimental::filesystem::path indexFile);