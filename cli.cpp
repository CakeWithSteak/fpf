#include "cli.h"
#include "modes.h"
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

std::optional<ModeInfo> tryGetMode(std::string str) {
    std::transform(str.cbegin(), str.cend(), str.begin(), [](char c){return std::tolower(c);});
    std::optional<ModeInfo> res = {};
    for(auto& [_, mode] : modes) {
        if(mode.cliName == str)
            res = mode;
    }
    return res;
}

std::string getExpression() {
    std::cout << "Expression> ";
    std::string expr;
    std::getline(std::cin, expr);
    return expr;
}

Options getOptions(int argc, char** argv) {
    po::options_description desc;

    desc.add_options()
       ("mode", po::value<std::string>(), "Fractal construction mode: fixed/fixed-dist/julia")
       ("expression", po::value<std::string>(), "The function to generate the fractal")
       ("width,w", po::value<int>()->default_value(1024), "Window width")
       ("height,h", po::value<int>()->default_value(1024), "Window height")
       ("refs-path", po::value<std::string>()->default_value("./refs.txt"), "References file path")
       ("metric-arg,m", po::value<float>(), "Metric argument")
       ("no-incremental-t", po::bool_switch(), "Disable incremental calculation of the line transform")
       ("double", po::bool_switch(), "Enable double precision mode");

    po::positional_options_description pos;
    pos.add("mode", 1);
    pos.add("expression", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
    po::notify(vm);

    Options opt;
    opt.width = vm["width"].as<int>();
    opt.height = vm["height"].as<int>();
    opt.refsPath = vm["refs-path"].as<std::string>();
    opt.forceDisableIncrementalLineTracing = vm["no-incremental-t"].as<bool>();
    opt.doublePrec = vm["double"].as<bool>();

    if(vm.count("metric-arg")) {
        opt.metricArg = vm["metric-arg"].as<float>();
    } else {
        opt.metricArg = {};
    }

    if(vm.count("mode")) {
        auto modeStr = vm["mode"].as<std::string>();
        auto mode = tryGetMode(modeStr);
        if(mode.has_value()) {
            opt.mode = *mode;
        } else if(std::filesystem::exists(modeStr)) {
            opt.deserializationPath = modeStr;
            return opt;
        } else {
            throw std::runtime_error("Invalid mode: \"" + modeStr + "\"");
        }
    } else {
        opt.mode = modes.at(FIXEDPOINT_ITERATIONS);
    }

    if(vm.count("expression"))
        opt.expression = vm["expression"].as<std::string>();
    else
        opt.expression = getExpression();

    return opt;
}