#! /bin/bash

python3 compute_images.py

cd ../../cmake-build-release

./clusterbuilder --index-directory=/mnt/WAREHOUSE2/Stash/SHREC2013_index --force-gpu=0 --quicci-dump-directory=/mnt/WAREHOUSE2/Stash/SHREC2013/descriptors/haystack


