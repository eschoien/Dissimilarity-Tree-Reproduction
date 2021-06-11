#pragma once

#include <projectSymmetry/types/Cluster.h>

Cluster* buildClusterFromDumpDirectory(const cluster::path &imageDumpDirectory,
                                       const cluster::path &indexDirectory,
                                       const unsigned int imagesPerBucket,
                                       bool forceCPU);