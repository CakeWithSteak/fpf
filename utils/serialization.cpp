#include "serialization.h"
#include "../Computation/shared_types.h"
#include <fstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <concepts>

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

// Serialize std::optional
namespace boost::serialization {
    template <class Archive, class T>
    void save(Archive& ar, const std::optional<T>& val, unsigned int version) {
        ar << val.has_value();
        ar << val.value_or(T());
    }

    template <class Archive, class T>
    void load(Archive& ar, std::optional<T>& val, unsigned int version) {
        bool hasValue;
        ar >> hasValue;
        T loadedVal;
        ar >> loadedVal;
        if(hasValue) {
            val = {loadedVal};
        }
    }

    template<class Archive, class T>
    void serialize(Archive& ar, std::optional<T>& t, const unsigned int version)
    {
        boost::serialization::split_free(ar, t, version);
    }

    // Serialize ShapeProps
    template <class Archive>
    void save(Archive& ar, const ShapeProps& val, unsigned int version) {
        ar << val.shape;
        switch (val.shape) {
            case LINE:
                ar << val.line.p1;
                ar << val.line.p2;
                break;
            case CIRCLE:
                ar << val.circle.center;
                ar << val.circle.r;
                break;
            default:
                throw std::runtime_error("Invalid tag on ShapeProps");
        }
    }

    template <class Archive>
    void load(Archive& ar, ShapeProps& val, unsigned int version) {
        ar >> val.shape;
        switch (val.shape) {
            case LINE:
                ar >> val.line.p1;
                ar >> val.line.p2;
                break;
            case CIRCLE:
                ar >> val.circle.center;
                ar >> val.circle.r;
                break;
            default:
                throw std::runtime_error("Invalid transform shape on deserialization");
        }
    }

    template<class Archive>
    void serialize(Archive& ar, ShapeProps& t, const unsigned int version)
    {
        boost::serialization::split_free(ar, t, version);
    }

    //Serialize HostFloatComplex and HostDoubleComplex
    template<class Archive, class T> requires std::same_as<T, HostFloatComplex> || std::same_as<T, HostDoubleComplex>
    void serialize(Archive& ar, T& val, const unsigned int version)
    {
        ar & val.re;
        ar & val.im;
    }
}