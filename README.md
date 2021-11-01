# Source code repository for the paper "Partial 3D Object Retrieval using Local Binary QUICCI Descriptors and Dissimilarity Tree Indexing"

-----

### Notice: This repository is for archival and reproduction purposes only. 
### Please refer to the following repositories for updated code and documentation:

[libShapeDescriptor](https://github.com/bartvbl/libShapeDescriptor)



-----

This repository contains:

- A reference implementation of the Dissimilarity Tree proposed in the paper.
- A reference implementation of the Modified Quick Intersection Count Change Image for partial object retrieval purposes.
- A script which can be used to completely reproduce all results presented in the paper.

## Instructions

You only need to have python 3 installed, which comes with Ubuntu. Any dependencies needed by the project itself can be installed using the menu system in the script itself.

You can run the script by executing:

```bash
python3 replicate.py
```

From the root of the repository.

Should the script fail due to missing dependencies, you can find shell scripts installing all necessary packages in the scripts/ directory.

Refer to the included Manual PDF for further instructions.

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

- Development and implementation: Bart Iver van Blokland, [NTNU Visual Computing Lab](https://www.idi.ntnu.no/grupper/vis/)
- Supervision: Theoharis Theoharis, [NTNU Visual Computing Lab](https://www.idi.ntnu.no/grupper/vis/)

If you use (parts of) this project in your research, we kindly ask you reference the papers on which this project is based:

    @article{van2021dissimilarity,
      title={Partial 3D Object Retrieval using Local Binary QUICCI Descriptors and Dissimilarity Tree Indexing},
      author={van Blokland, Bart Iver and Theoharis, Theoharis},
      journal={Computers \& Graphics},
      volume="100",
      pages="32--42",
      year={2021},
      publisher={Elsevier}
    }
    
    @article{van2020indexing,
      title={An Indexing Scheme and Descriptor for 3D Object Retrieval Based on Local Shape Querying},
      author={van Blokland, Bart Iver and Theoharis, Theoharis},
      journal={Computers \& Graphics},
	  volume="92",
	  pages="55--66",
      year={2020},
      publisher={Elsevier}
    }

