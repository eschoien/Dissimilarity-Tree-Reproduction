#include <iostream>
#include <algorithm>
#include <vector>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>
#include "Signature.h"
#include "Permutation.h"

// want to pass pointer instead of returning
void computeDescriptorSignature(ShapeDescriptor::QUICCIDescriptor descriptor, std::vector<int>* signatures, std::vector<std::vector<int>> permutations) {
// std::vector<int> computeDescriptorSignature(ShapeDescriptor::QUICCIDescriptor descriptor, std::vector<std::vector<int>> permutations) {

    for (int p = 0; p < permutations.size(); p++) {

        std::vector<int> permutation = permutations[p];

        // improve this condition to be based on spinImagePixels etc.
        // perhaps change to for loop

        int m = 0;
        while (m < 1024) {
            // make this line more readable?
            if (descriptor.contents[(permutation[m] / 32)] & (1 << (permutation[m] % 32)))
            {
                break;
            }
            m++;
        }
        // m is now the signature of this descriptor for the current permutation
        signatures->push_back(m);
        
    }
}