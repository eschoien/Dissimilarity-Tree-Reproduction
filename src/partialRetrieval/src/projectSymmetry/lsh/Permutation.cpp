#include "Permutation.h"
#include <algorithm>
#include <vector>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/free/array.h>

/* 
Should add "random-seed" as a parameter, to be used in the random shuffle
Ensure that the random permutations are replicable
*/

std::vector<std::vector<int>> create_permutations(int numberOfPermutations) {

    std::vector<std::vector<int>> permutations;

    for (int n = 0; n < numberOfPermutations; n++) {

        std::vector<int> numbers;

        for (int i=0; i <= 1023; i++) {
            numbers.push_back(i);
        }

        std::random_shuffle(&numbers[0], &numbers[1024]);

        permutations.push_back(numbers);
        
        // do we need to free anything?
        // free(numbers);
    }

    return permutations;
}