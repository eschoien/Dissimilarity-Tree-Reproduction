# Source code repository for the paper ""

-----

This repository contains:

- A reference implementation of the LSH-based pipeline proposed in the paper.
- A reference implementation of our Directionally Modified QUICCI descriptor.

The modifications relevant to our master's thesis are in the files:
- This contains our modifications to the QUICCI descriptors on lines 302-372:
	- src/libShapeDescriptor/src/shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cu

- These contains our LSH-based partial retrieval pipeline:
	- src/partialRetrieval/src/projectSymmetry/lsh/
	- src/partialRetrieval/tools/
		- hashTableBuilder/
		- hashTableSearcher/
		- signatureSearcher/

## Instructions

You only need to have python 3 installed, which comes with Ubuntu. Any dependencies needed by the project itself can be installed using the menu system in the script itself.

You can run the script by executing:

```bash
python3 replicate.py
```

From the root of the repository.

Should the script fail due to missing dependencies, you can find shell scripts installing all necessary packages in the scripts/ directory.

Refer to the included Manual PDF for further instructions.

The options which were created for this master's thesis are options 16-23. 

## System Requirements

The RAM and Disk space requirements are only valid when attempting to reproduce the presented results.

The codebase _should_ be able to compile on Windows, but due to some CUDA driver/SDK compatbility issues we have not yet been able to verify this.

Type | Requirements
-----|----------------------------------------------------------------------------
CPU  | Does not matter
RAM  | At least 64GB
Disk | Must have about ~120GB of storage available to store the downloaded datasets
GPU  | Any NVIDIA GPU (project uses CUDA)
OS   | Ubuntu 16 or higher. Project has been tested on 18 and 20.

## Credits

This repository is a fork from Bart Iver van Blokland (https://github.com/bartvbl/Dissimilarity-Tree-Reproduction)
