#pragma once

#include <experimental/filesystem>

namespace cluster {
    typedef std::experimental::filesystem::path path;
    namespace filesystem = std::experimental::filesystem;
}