#pragma once

#include <iostream>
#include <vector>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>

namespace lsh {
    namespace print {
        inline void signature(std::vector<int> signature) {
            for (int s = 0; s < signature.size(); s++) {
                std::cout << signature[s];
                if (s < signature.size()-1) {
                    std::cout << " "; 
                }
            }
            std::cout << std::endl;
        }
    }
}

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
    int objectCount;
    std::vector<std::vector<int>> permutations;
};

void computeDescriptorSignature(ShapeDescriptor::QUICCIDescriptor descriptor, std::vector<int>* descriptorSignaturesPtr, std::vector<std::vector<int>> permutations);