#pragma once

#include <vector>

struct DescriptorSignature {
    unsigned int file_id;
    unsigned int descriptor_id;
    std::vector<int> signatures;
};