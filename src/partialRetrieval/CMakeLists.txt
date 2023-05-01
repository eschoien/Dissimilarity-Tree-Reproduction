#cmake_policy(SET CMP0048 NEW)

set(CMAKE_CXX_STANDARD 14)

cmake_minimum_required(VERSION 3.0)
project(ProjectSymmetry LANGUAGES CXX CUDA)


add_subdirectory(../libShapeDescriptor ${CMAKE_CURRENT_BINARY_DIR}/libShapeDescriptor)

find_package(CUDA REQUIRED)
find_package(OpenMP)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#
# GLFW options
#
option (GLFW_INSTALL OFF)
option (GLFW_BUILD_DOCS OFF)
option (GLFW_BUILD_EXAMPLES OFF)
option (GLFW_BUILD_TESTS OFF)
#add_subdirectory (lib/glfw)

# Library contains GLFW:
option(PMP_BUILD_TESTS OFF)
option(PMP_BUILD_DOCS OFF)
option(PMP_BUILD_EXAMPLES OFF)
add_subdirectory(lib/pmp-library ${CMAKE_CURRENT_BINARY_DIR}/pmp-library)


# Utility for baking in the current git commit into source code
set(PRE_CONFIGURE_FILE "lib/cmake-git/git.cc.in")
set(POST_CONFIGURE_FILE "${CMAKE_CURRENT_BINARY_DIR}/git.cc")
include(lib/cmake-git/git_watcher.cmake)
add_library(gitinfo STATIC ${POST_CONFIGURE_FILE})
target_include_directories(gitinfo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(gitinfo check_git)

include_directories (
        src/
        lib/glm/
        lib/glad/include/
        lib/glfw/include/
        lib/nvidia-samples-common/
        lib/lodepng/
        lib/arrrgh/
        lib/json/
        lib/fast-lzma2/src/
        lib/
        lib/cmake-git/
        lib/pmp-library/external/eigen
        ../libShapeDescriptor/src
        ${CUDA_INCLUDE_DIRS}
)


file (GLOB_RECURSE LIBRARY_HEADERS             src/*.hpp
        src/*.h
        src/*.cuh)
file (GLOB_RECURSE LIBRARY_KERNELS             src/*.cu
        lib/nvidia-samples-common/*.cu)
file (GLOB_RECURSE LIBRARY_SOURCES             src/*.cpp
        src/*.cxx
        src/*.cc
        src/*.c
        lib/glad/src/glad.cpp
        lib/lodepng/lodepng.cpp)
file (GLOB_RECURSE TEST_SOURCES                tests/*.cpp)

add_definitions (-DGLFW_INCLUDE_NONE)
add_library(projectsymmetry STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS} ${LIBRARY_KERNELS})
target_link_libraries(projectsymmetry glfw ShapeDescriptor stdc++fs fast-lzma2 gitinfo)
target_include_directories(projectsymmetry PUBLIC src/)

add_executable(runProjectSymmetryTests ${TEST_SOURCES})
target_link_libraries(runProjectSymmetryTests projectsymmetry ShapeDescriptor Catch2::Catch2)

add_executable(clusterbuilder tools/clusterbuilder/main.cpp)
target_link_libraries(clusterbuilder glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2)

add_executable(simplesearch tools/simplesearch/main.cpp)
target_link_libraries(simplesearch glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 gitinfo)

add_executable(descriptorDumper tools/descriptorDumper/main.cpp)
target_link_libraries(descriptorDumper glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2)

add_executable(alignmentexperiment tools/alignmentExperiment/main.cpp)
target_link_libraries(alignmentexperiment glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2)

add_executable(querysetgenerator tools/querysetgenerator/main.cpp)
target_link_libraries(querysetgenerator glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp)

add_executable(smartquiccirenderer tools/smartquiccirenderer/main.cpp)
target_link_libraries(smartquiccirenderer glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp)

add_executable(sequentialSearchBenchmark tools/sequentialClusterSearcher/main.cpp)
target_link_libraries(sequentialSearchBenchmark glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(indexedSearchBenchmark tools/indexedClusterSearcher/main.cpp)
target_link_libraries(indexedSearchBenchmark glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(areaCalculator tools/areaCalculator/main.cpp)
target_link_libraries(areaCalculator glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(indexWalker tools/indexWalker/main.cpp)
target_link_libraries(indexWalker glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(objectSearch tools/objectSearch/main.cpp)
target_link_libraries(objectSearch glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(occurrenceCounter tools/occurrenceCounter/main.cpp)
target_link_libraries(occurrenceCounter glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(remeshedCorrespondenceFinder tools/remeshedCorrespondenceFinder/main.cpp)
target_link_libraries(remeshedCorrespondenceFinder glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(edgeRemovalExperiment
        tools/edgeRemovalExperiment/main.cpp
        tools/edgeRemovalExperiment/Histogram.cpp
        tools/edgeRemovalExperiment/Histogram.h)
target_link_libraries(edgeRemovalExperiment glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(signatureBuilder tools/signatureBuilder/main.cpp)
target_link_libraries(signatureBuilder glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(descriptorSignatureTest tools/descriptorSignatureTest/main.cpp)
target_link_libraries(descriptorSignatureTest glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(signatureMatchingTest tools/signatureMatchingTest/main.cpp)
target_link_libraries(signatureMatchingTest glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(signatureSearcher tools/signatureSearcher/main.cpp)
target_link_libraries(signatureSearcher glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(hashtableBuilder tools/hashtableBuilder/main.cpp)
target_link_libraries(hashtableBuilder glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)

add_executable(hashtableSearcher tools/hashtableSearcher/main.cpp)
target_link_libraries(hashtableSearcher glfw projectsymmetry ShapeDescriptor stdc++fs fast-lzma2 pmp gitinfo)