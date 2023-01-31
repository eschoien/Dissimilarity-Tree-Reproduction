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

void writeSignatures(ObjectSignature objectSig, const std::experimental::filesystem::path outputDirectory);
// ObjectSignature* readSignature(std::experimental::filesystem::path outputFile);