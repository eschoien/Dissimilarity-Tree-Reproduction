sed -i 's/^#define spinImageWidthPixels .*/#define spinImageWidthPixels 64/' ../../libShapeDescriptor/src/shapeDescriptor/libraryBuildSettings.h
make -j
