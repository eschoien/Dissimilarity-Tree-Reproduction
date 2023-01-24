#include <iostream>
#include <algorithm>
#include <vector>

std::vector<std::vector<int>> create_permutations(int numberOfPermutations) {

    std::vector<std::vector<int>> permutations;

    for (int n = 0; n < numberOfPermutations; n++) {

        std::vector<int> numbers;

        for (int i=1; i <= 1024; i++) {
            numbers.push_back(i);
        }

        std::random_shuffle(&numbers[0], &numbers[1024]);

        //std::cout << numbers[0] << " " << numbers[1] << " " <<  numbers[2] << "\n\n";

        permutations.push_back(numbers);
        
    }

    return permutations;
}

int main() {

    int n = 10;

    std::vector<std::vector<int>> permutations = create_permutations(n);

    std::cout << permutations.size() << "\n";
    std::cout << permutations[0].size() << "\n";

    for (int i = 0; i < n; i++) {

        std::cout << permutations[i][0] << " " << permutations[i][1] << " " <<  permutations[i][2] << "\n\n";
    }

    return 0;
}
