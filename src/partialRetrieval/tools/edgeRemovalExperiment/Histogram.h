#pragma once

#include <vector>

struct Histogram {
    struct Bin {
        double min = 0;
        double max = 0;
        size_t count = 0;
    };

    double min;
    double max;
    double step;
    unsigned int binCount;

    std::vector<Bin> contents;

    Histogram(double min, double max, unsigned int binCount)
        : min(min), max(max), step((max - min) / double(binCount)), binCount(binCount) {
        contents.resize(binCount);

        for(unsigned int i = 0; i < binCount; i++) {
            contents.at(i).count = 0;
            contents.at(i).min = min + step * double(i);
            contents.at(i).max = min + step * double(i + 1);
        }
        contents.at(binCount - 1).max = max;
    }

    void count(double value);

};