#include "cli.h"
#include "modes.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <complex>

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

template <typename T>
void addAnimParam(Interpolate<T>& target, T start, T end) {
    target = Interpolate<T>(start, end);
}

template <typename T>
void addAnimParam(Interpolate<T>& target, const po::variables_map& vm, T start, const char* name) {
    if(vm.count(name))
        addAnimParam(target, start, vm[name].as<T>());
    else
        addAnimParam(target, start, start);
}

template <typename T>
void addAnimParam(std::optional<Interpolate<T>>& target, const po::variables_map& vm, T start, const char* name) {
    if(vm.count(name))
        target.emplace(Interpolate<T>(start, vm[name].as<T>()));
}

template <typename T>
void addAnimParam(Interpolate<T>& target, const po::variables_map& vm, const char* startName, const char* endName) {
    addAnimParam(target, vm, vm.at(startName).as<T>(), endName);
}

template <typename T>
void addAnimParam(std::optional<Interpolate<T>>& target, const po::variables_map& vm, const char* startName, const char* endName) {
    if(vm.count(startName))
        addAnimParam(target, vm, vm[startName].as<T>(), endName);
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
       ("mode", po::value<std::string>(), "Fractal construction mode")
       ("expression", po::value<std::string>(), "The function to generate the fractal")
       ("width,w", po::value<int>()->default_value(1024), "Window width")
       ("height,h", po::value<int>()->default_value(1024), "Window height")
       ("refs-path", po::value<std::string>()->default_value("./refs.txt"), "References file path")
       ("metric-arg,m", po::value<double>(), "Metric argument")
       ("no-incremental-t", po::bool_switch(), "Disable incremental calculation of the line transform")
       ("double", po::bool_switch(), "Enable double precision mode")
       ("param,p", po::value<std::complex<double>>()->default_value(0), "Parameter")
       ("center", po::value<std::complex<double>>()->default_value(0), "View center")
       ("zoom", po::value<double>()->default_value(4), "The side length of the viewport")
       ("max-iters,i", po::value<int>(), "Maximum iterations")
       ("color-cutoff", po::value<double>(), "The maximum value to color")
       ("anim-length,A", po::value<double>(), "The length of the animation in seconds")
       ("anim-path,o",  po::value<std::filesystem::path>(), "The directory to export animation frames into")
       ("anim-fps", po::value<int>()->default_value(60), "Animation FPS")
       ("anim-max-iters-end", po::value<int>())
       ("anim-metric-arg-end", po::value<double>())
       ("anim-p-end", po::value<std::complex<double>>())
       ("anim-center-end", po::value<std::complex<double>>())
       ("anim-zoom-end", po::value<double>())
       ("anim-color-cutoff-end", po::value<double>())
       ("anim-path-start", po::value<std::complex<double>>())
       ("anim-path-end", po::value<std::complex<double>>())
       ("anim-line-trans-a-start", po::value<std::complex<double>>())
       ("anim-line-trans-a-end", po::value<std::complex<double>>())
       ("anim-line-trans-b-start", po::value<std::complex<double>>())
       ("anim-line-trans-b-end", po::value<std::complex<double>>())
       ("anim-line-trans-iters-start", po::value<int>()->default_value(0))
       ("anim-line-trans-iters-end", po::value<int>())
       ("anim-background,H", po::bool_switch()->default_value(false), "Creates animation without opening a window")
       ;

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
    opt.p = vm["param"].as<std::complex<double>>();
    opt.viewportCenter = vm["center"].as<std::complex<double>>();
    opt.viewportBreadth = vm["zoom"].as<double>() / 2;
    opt.animBackground = vm["anim-background"].as<bool>();

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

    if(vm.count("metric-arg"))
        opt.metricArg = vm["metric-arg"].as<double>();
    else
        opt.metricArg = opt.mode.argInitValue;

    if(vm.count("max-iters"))
        opt.maxIters = vm["max-iters"].as<int>();
    else
        opt.maxIters = opt.mode.initMaxIters;

    if(vm.count("expression"))
        opt.expression = vm["expression"].as<std::string>();
    else
        opt.expression = getExpression();

    if(vm.count("color-cutoff"))
        opt.colorCutoff = vm["color-cutoff"].as<double>();
    else if(opt.mode.defaultColorCutoff != -1)
        opt.colorCutoff = opt.mode.defaultColorCutoff;

    if(vm.count("anim-length") && vm.count("anim-path")) {
        AnimationParams anim;
        anim.duration = vm["anim-length"].as<double>();
        anim.fps = vm["anim-fps"].as<int>();
        opt.animPath = vm["anim-path"].as<std::filesystem::path>();

        addAnimParam(anim.maxIters, vm, opt.maxIters, "anim-max-iters-end");
        addAnimParam(anim.metricArg, vm, opt.metricArg, "anim-metric-arg-end");
        addAnimParam(anim.p, vm, opt.p, "anim-p-end");
        addAnimParam(anim.viewportCenter, vm, opt.viewportCenter, "anim-center-end");

        if(vm.count("anim-zoom-end"))
            addAnimParam(anim.viewportBreadth, opt.viewportBreadth, vm["anim-zoom-end"].as<double>() / 2);
        else
            addAnimParam(anim.viewportBreadth, opt.viewportBreadth, opt.viewportBreadth);

        addAnimParam(anim.colorCutoff, vm, *opt.colorCutoff, "anim-color-cutoff-end");
        addAnimParam(anim.pathStart, vm, "anim-path-start", "anim-path-end");
        addAnimParam(anim.lineTransIteration, vm, "anim-line-trans-iters-start", "anim-line-trans-iters-end");

        if(vm.count("anim-line-trans-a-start") && vm.count("anim-line-trans-b-start") &&
         vm.count("anim-line-trans-a-end") && vm.count("anim-line-trans-b-end") ) {
            addAnimParam(anim.lineTransStart, vm, "anim-line-trans-a-start", "anim-line-trans-a-end");
            addAnimParam(anim.lineTransEnd, vm, "anim-line-trans-b-start", "anim-line-trans-b-end");
        }

        opt.animParams = std::move(anim);
    }

    return opt;
}