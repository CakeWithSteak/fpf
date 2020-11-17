#include "findCuda.h"
#include <cstdlib>

namespace fs = std::filesystem;

fs::path findCudaIncludePath() {
    char* envpath = std::getenv("CUDA_PATH");
    if(envpath) {
        auto path = fs::path(envpath) / "include";
        if (fs::exists(path))
            return path;
    }

#if defined(unix) || defined(__unix) || defined(__unix__)
    if(fs::exists("/usr/local/cuda/include/"))
        return "/usr/local/cuda/include/";
    const fs::path basedir = "/usr/local";
    const std::string versionPrefix = "cuda-";
#elif defined(_MSC_VER) || defined(__WIN32)
    char* pfilesenv = std::getenv("PROGRAMFILES");
    fs::path pfiles = pfilesenv && fs::exists(pfilesenv) ? pfilesenv : "C:/Program Files";
    const fs::path basedir = pfiles / "NVIDIA GPU Computing Toolkit" / "CUDA";
    const std::string versionPrefix = "v";
#else
#error "Unsupported operating system."
#endif


    if(fs::exists(basedir)) {
        for(auto& f : fs::directory_iterator(basedir)) {
            if(f.path().filename().string().starts_with(versionPrefix) && f.is_directory() && fs::exists(f.path() / "include"))
                return f.path() / "include";
        }
    }
    throw std::runtime_error("Couldn't find CUDA path. Make sure that CUDA is installed correctly, "
                             "and try setting the CUDA_PATH environment variable, or provide the CUDA path manually using the --cuda-path option.");
};