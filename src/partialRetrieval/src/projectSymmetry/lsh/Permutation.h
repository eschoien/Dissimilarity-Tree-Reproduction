#pragma once

#include <vector>
#include <iostream>

/*
struct Permutation {
    std::vector<int> indices;
};

struct Permutations {
    std::vector<Permutation> permutations;
};
*/

namespace lsh {
    namespace print {

        inline void permutation(std::vector<int> permutation, unsigned int number) {
            //number = std::min(number, permutation.size());
            for (int i = 0; i < number; i++) {
                std::cout << permutation[i];
                if (i < number - 1) {
                    std::cout << " ";
                }
            }
        }

        inline void permutations(std::vector<std::vector<int>> permutations, unsigned int elementsPerPermutation) {
            for (int i = 0; i < permutations.size(); i++) {
                std::cout << i << ":  ";
                lsh::print::permutation(permutations[i], elementsPerPermutation);
                std::cout << std::endl;
            }
        }   
    }
}

std::vector<std::vector<int>> create_permutations(int numberOfPermutations, size_t seed);