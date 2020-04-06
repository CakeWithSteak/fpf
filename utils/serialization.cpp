#include "serialization.h"
#include <fstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

void serialize(const State& data, const std::filesystem::path& path) {
    std::ofstream file(path);
    boost::archive::text_oarchive oa(file);
    oa << data;
}

State deserialize(const std::filesystem::path& path) {
    State data;

    std::ifstream file(path);
    boost::archive::text_iarchive ia(file);
    ia >> data;
    return data;
}

void save(State& state) {
    std::cout << "Save to> ";
    std::filesystem::path filename;
    std::cin >> filename;
    serialize(state, filename);
}