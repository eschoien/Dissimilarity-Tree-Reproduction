# Source code repository for the paper ""

-----

**Below are the instructions and system requirements of the original Dissimilarity Tree reproduction repository.**

> ## Instructions (original)
> 
> You only need to have python 3 installed, which comes with Ubuntu. Any dependencies needed by the project itself can be installed using the menu system in the script itself.
>
> You can run the script by executing:
>
> ```bash
> python3 replicate.py
> ```
>
> From the root of the repository.
>
> Should the script fail due to missing dependencies, you can find shell scripts installing all necessary packages in the scripts/ directory.
>
> Refer to the included Manual PDF for further instructions.
>
> ## System Requirements (original)
>
> The RAM and Disk space requirements are only valid when attempting to reproduce the presented results.
> 
> The codebase _should_ be able to compile on Windows, but due to some CUDA driver/SDK compatbility issues we have not yet been able to verify this.
> 
> Type | Requirements
> -----|----------------------------------------------------------------------------
> CPU  | Does not matter
> RAM  | At least 64GB
> Disk | Must have about ~120GB of storage available to store the downloaded datasets
> GPU  | Any NVIDIA GPU (project uses CUDA)
> OS   | Ubuntu 16 or higher. Project has been tested on 18 and 20.
> 

# Overview and Instructions for this thesis project

## Overview

This repository is a fork of the Dissimilarity-Tree-Reproduction, which extends it with the following:

- The implementation of the LSH-based pipeline proposed in the thesis.
- The implementation of the Directionally Modified QUICCI descriptor proposed in the thesis.

The modifications for our master's thesis project are:

- Modifications for the directional QUICCI descriptors:
	- src/libShapeDescriptor/src/shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cu
		- Lines 302-372:
	- src/libShapeDescriptor/src/libraryBuildSettings.h
		- Line 22 changes the direction of the QUICCI descriptor
		- Requires recompiling and recomputing of descriptors and dissimilarity tree

- These contains our LSH-based partial retrieval pipeline:
	- src/partialRetrieval/src/projectSymmetry/lsh/*
	- src/partialRetrieval/tools/
		- hashTableBuilder/main.cpp
		- hashTableSearcher/main.cpp
		- signatureBuilder/main.cpp
		- signatureSearcher/main.cpp

## Instructions

**Options 16-23 in the replicate.py script were created as a part of this thesis.**

After cloning the project, make sure to:
1. Download the SHREC2016 dataset and the precomputed augmented SHREC2016 dataset using the replicate.py script
which is required in all of the cases below.

### Testing the QUICCI directions:

1. Choose between horizontal, vertical or combined QUICCI as specified in the Overview above.
2. Using the replicate.py script (need to redo this each time / for each direction):
	- Compile the project
	- Compute the descriptors
	- Compute the dissimilarity tree
3. Run option 11: dissimilarity tree execution time evalution: compute entire chart

### Testing LSH-based partial retrieval pipeline:
1. (make sure that the original horizontal QUICCI descriptor is used as described above)
2. Compile the project
3- Compute (or download the precomputed) descriptors
4. Run option 20, signature experiment, which will produce a measurement file for every combination of parameters

### Testing the LSH-based partial retrieval pipeline using hash tables:
1. (make sure that the original horizontal QUICCI descriptor is used as described above)
2. Run option 23: hashtable searcher

### Top-k measurements for the dissimilarity tree
- Compute or download the precomputed dissimilarity tree
- Run option 21: top-k results for dissimilarity tree

**Various scripts, for creating the figures and tables from the measurement files produced, can be found in analysis/master_project directory in the lsh-experiment branch. These are not included in the master branch as they are created ad-hoc for visualizing the results.**

## Credits

This repository is a fork from Bart Iver van Blokland (https://github.com/bartvbl/Dissimilarity-Tree-Reproduction)
