#include "cli.h"
#include "modes.h"
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

std::istream& operator>>(std::istream& in, ModeInfo& metric) {
    std::string str;
    in >> str;
    std::transform(str.cbegin(), str.cend(), str.begin(), [](char c){return std::tolower(c);});
    //todo refactor
    if(str == "fixed")
        metric = modes.at(FIXEDPOINT_ITERATIONS);
    else if(str == "fixed-dist")
        metric = modes.at(FIXEDPOINT_EUCLIDEAN);
    else if(str == "julia")
        metric = modes.at(JULIA);
    else if(str == "displacement")
        metric = modes.at(VECTORFIELD_MAGNITUDE);
    else
        in.setstate(std::ios::failbit);
    return in;
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
       ("mode", po::value<ModeInfo>(), "Fractal construction mode: fixed/fixed-dist/julia")
       ("expression", po::value<std::string>(), "The function to generate the fractal")
       ("width,w", po::value<int>()->default_value(1024), "Window width")
       ("height,h", po::value<int>()->default_value(1024), "Window height");

    po::positional_options_description pos;
    pos.add("mode", 1);
    pos.add("expression", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
    po::notify(vm);

    Options opt;
    opt.width = vm["width"].as<int>();
    opt.height = vm["height"].as<int>();

    if(vm.count("mode"))
        opt.mode = vm["mode"].as<ModeInfo>();
    else
        opt.mode = modes.at(FIXEDPOINT_ITERATIONS);

    if(vm.count("expression"))
        opt.expression = vm["expression"].as<std::string>();
    else
        opt.expression = getExpression();

    return opt;
}