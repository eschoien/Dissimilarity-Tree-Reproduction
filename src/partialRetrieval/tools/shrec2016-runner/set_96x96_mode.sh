sed -i 's/^#define spinImageWidthPixels .*/#define spinImageWidthPixels 96/' ../../libShapeDescriptor/src/shapeDescriptor/libraryBuildSettings.h 
make -j time 
