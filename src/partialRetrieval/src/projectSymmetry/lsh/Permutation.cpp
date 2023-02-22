#include "Permutation.h"
#include <algorithm>
#include <vector>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <random>

/* 
Should add "random-seed" as a parameter, to be used in the random shuffle
Ensure that the random permutations are replicable
*/

std::vector<std::vector<int>> create_permutations(int numberOfPermutations, size_t seed) {

    std::random_device rd("/dev/urandom");
    size_t randomSeed = seed != 0 ? seed : rd();
    std::minstd_rand0 generator{randomSeed};
    std::uniform_real_distribution<float> distribution(0, 1);

    std::vector<std::vector<int>> permutations;

    for (int n = 0; n < numberOfPermutations; n++) {

        std::vector<int> numbers;

        for (int i=0; i <= 1023; i++) {
            numbers.push_back(i);
        }

        std::shuffle(&numbers[0], &numbers[1024], generator);

        permutations.push_back(numbers);
        
        // do we need to free anything?
        // free(numbers);
    }

    return permutations;
}