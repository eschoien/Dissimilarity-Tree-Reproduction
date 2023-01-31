#pragma once

#include <vector>

struct DescriptorSignature {
    unsigned int descriptor_id;
    std::vector<int> signatures;
};

struct ObjectSignature {
    unsigned int file_id;
    std::vector<DescriptorSignature> descriptorSignatures;
};