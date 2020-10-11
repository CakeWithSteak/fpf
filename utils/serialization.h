#pragma once
#include <filesystem>
#include <boost/serialization/access.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/serialization/string.hpp>
#include "State.h"

void save(State& state);
State deserialize(const std::filesystem::path& path);