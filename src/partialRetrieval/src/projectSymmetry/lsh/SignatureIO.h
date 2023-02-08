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

// can we make read and write independent of number of permutations?
void writeSignatures(ObjectSignature objectSig, const std::experimental::filesystem::path outputDirectory, const unsigned int numberOfPermutations);
ObjectSignature *readSignature(const std::experimental::filesystem::path indexFile, const unsigned int numberOfPermutations);

void writeSignatureIndex(SignatureIndex sigIndex, const std::experimental::filesystem::path outputFile);
SignatureIndex *readSignatureIndex(const std::experimental::filesystem::path indexFile);