#pragma once

#include <projectSymmetry/types/Cluster.h>
#include "DiskBasedImageRegistry.h"

void writeCluster(Cluster* cluster, cluster::path outputFile);
Cluster* readCluster(cluster::path outputFile);