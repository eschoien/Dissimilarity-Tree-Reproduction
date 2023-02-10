#include <iostream>
#include <vector>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include "Signature.h"
#include "Permutation.h"

void computeDescriptorSignature(ShapeDescriptor::QUICCIDescriptor descriptor, std::vector<int>* signaturePtr, std::vector<std::vector<int>> permutations) {
    
    for (int p = 0; p < permutations.size(); p++) {

        std::vector<int> permutation = permutations[p];
        int m = 0;
        bool quicciBit = false;
        
        // improve this condition to be based on spinImagePixels etc.
        while (m < 1024) {
            // Old one-line version: (descriptor.contents[(permutation[m] / 32)] & (1 << (permutation[m] % 32)))
            unsigned int quicciRow = descriptor.contents[(permutation[m] / 32)];
            unsigned int quicciCol = permutation[m] % 32;
            quicciBit = quicciRow & (1 << quicciCol);
            if (quicciBit) {
                break;
            }
            m++;
        }
        // m is now the signature of this descriptor for the current permutation
        signaturePtr->push_back(m);
    }
}

double computeJaccardSimilarity(std::vector<int> signature1, std::vector<int> signature2) {

    unsigned int matchScore = 0;

    for (unsigned int k = 0; k < signature1.size(); k++) {
        if (signature1[k] == signature2[k]) {
            matchScore++;
        }
    }

    return (double) matchScore / signature1.size();
}