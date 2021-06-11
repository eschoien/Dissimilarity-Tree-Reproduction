#include <stdexcept>
#include "Histogram.h"

void Histogram::count(double value) {
    if(value < min || value > max) {
        throw std::runtime_error("Attempted to count histogram value of " + std::to_string(value) + ", which is not in the range (" + std::to_string(min) + ", " + std::to_string(max) + ").");
    }

    unsigned int binIndex = (unsigned int)(((value - min) / (max - min)) * double(binCount));

    // Handling the edge case when value == max
    binIndex = std::min<unsigned int>(binIndex, binCount - 1);

    #pragma omp atomic
    contents.at(binIndex).count++;
}
