#pragma once

#include <vector>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>

struct DescriptorSignature {
    unsigned int descriptor_id;
    std::vector<int> signatures;
};

struct ObjectSignature {
    unsigned int file_id;
    std::vector<DescriptorSignature> descriptorSignatures;
};

/* Signature metadata, minhash permutations etc. (add more here) */
struct SignatureIndex {
    std::vector<ObjectSignature> objectSignatures;
    std::vector<std::vector<int>> permutations;
};

void computeDescriptorSignature(ShapeDescriptor::QUICCIDescriptor descriptor, std::vector<int>* descriptorSignaturesPtr, std::vector<std::vector<int>> permutations);