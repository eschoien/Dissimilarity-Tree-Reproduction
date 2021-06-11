#pragma once

struct ClusterEntry {
    // To save space, we only store the index of the file where the entry originated from.
    // This is translated to a full file path based on the main file list in Index.
    unsigned int fileIndex;

    // Within the object, this is the image index that this bucket entry refers to.
    unsigned int imageIndex;

    ClusterEntry(unsigned int fileIndex, unsigned int imageIndex) :
    fileIndex(fileIndex),
    imageIndex(imageIndex) {}

    // Default constructor to allow std::vector resizing
    ClusterEntry() : fileIndex(0), imageIndex(0) {}

    bool operator< (const ClusterEntry& rhs) const {
        if(fileIndex != rhs.fileIndex) {
            return fileIndex < rhs.fileIndex;
        }

        if(imageIndex != rhs.imageIndex) {
            return imageIndex < rhs.imageIndex;
        }

        return false;
    }

    bool operator==(const ClusterEntry& other) const {
        return fileIndex == other.fileIndex && imageIndex == other.imageIndex;
    }
};