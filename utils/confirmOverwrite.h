#pragma once
#include <filesystem>
#include <iostream>
#include <cctype>

// Confirms that the provided file doesn't exist, or that the user agrees to overwrite it.
inline bool confirmOverwrite(const std::filesystem::path& path) {
    if(std::filesystem::exists(path)) {
        std::cout << "File " << path << " already exists. Are you sure you want to overwrite it? (y/n) ";
        char response;
        std::cin >> response;
        std::cin.ignore(30000, '\n');
        return std::tolower(response) == 'y';
    }
    return true;
}