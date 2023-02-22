#pragma once

#include <arrrgh.hpp>
#include <iostream>
#include <shapeDescriptor/utilities/fileutils.h>
#include "Signature.h"

SignatureIndex buildSignaturesFromDumpDirectory(const std::experimental::filesystem::path &imageDumpDirectory, const std::experimental::filesystem::path &outputDirectory, const unsigned int numberOfPermutations, size_t seed);