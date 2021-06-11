cd ../../cmake-build-release/

mkdir /mnt/WAREHOUSE2/Stash/SHREC2016_index_32x32
mkdir /mnt/WAREHOUSE2/Stash/SHREC2016_index_64x64
mkdir /mnt/WAREHOUSE2/Stash/SHREC2016_index_96x96

sed -i 's/^#define spinImageWidthPixels .*/#define spinImageWidthPixels 32/' ../../libShapeDescriptor/src/shapeDescriptor/libraryBuildSettings.h
make -j
time ./clusterbuilder --index-directory=/mnt/WAREHOUSE2/Stash/SHREC2016_index_32x32 --quicci-dump-directory=/mnt/WAREHOUSE2/Stash/SHREC2016_32x32/haystack_32x32 --force-gpu=0


sed -i 's/^#define spinImageWidthPixels .*/#define spinImageWidthPixels 64/' ../../libShapeDescriptor/src/shapeDescriptor/libraryBuildSettings.h
make -j
time ./clusterbuilder --index-directory=/mnt/WAREHOUSE2/Stash/SHREC2016_index_64x64 --quicci-dump-directory=/mnt/WAREHOUSE2/Stash/SHREC2016_64x64/haystack_64x64 --force-gpu=0


sed -i 's/^#define spinImageWidthPixels .*/#define spinImageWidthPixels 96/' ../../libShapeDescriptor/src/shapeDescriptor/libraryBuildSettings.h
make -j
time ./clusterbuilder --index-directory=/mnt/WAREHOUSE2/Stash/SHREC2016_index_96x96 --quicci-dump-directory=/mnt/WAREHOUSE2/Stash/SHREC2016_96x96/haystack_96x96 --force-gpu=0 --force-cpu
