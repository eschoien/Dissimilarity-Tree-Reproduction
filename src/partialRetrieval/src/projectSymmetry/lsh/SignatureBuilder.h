#pragma once

#include <arrrgh.hpp>
#include <iostream>
#include <shapeDescriptor/utilities/fileutils.h>

std::vector<DescriptorSignature> buildSignaturesFromDumpDirectory(const std::experimental::filesystem::path &imageDumpDirectory, const std::experimental::filesystem::path &outputDirectory, const unsigned int number_of_permutations);