sed -i 's/^#define spinImageWidthPixels .*/#define spinImageWidthPixels 32/' ../../libShapeDescriptor/src/shapeDescriptor/libraryBuildSettings.h
make -j
