#include "cli.h"
#include "modes.h"
#include "utils/findCuda.h"
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
       ("refs-path", po::value<std::string>(), "References file path")
       ("metric-arg,m", po::value<double>(), "Metric argument")
       ("no-incremental-t", po::bool_switch(), "Disable incremental calculation of the line transform")
       ("double", po::bool_switch(), "Enable double precision mode")
       ("param,p", po::value<std::complex<double>>()->default_value(0), "Parameter")
       ("center", po::value<std::complex<double>>()->default_value(0), "View center")
       ("zoom", po::value<double>()->default_value(4), "The side length of the viewport")
       ("max-iters,i", po::value<int>(), "Maximum iterations")
       ("color-cutoff", po::value<double>(), "The maximum value to color")
       ("anim-length,A", po::value<int>(), "The number of frames to animate")
       ("anim-path,o",  po::value<std::filesystem::path>(), "The directory to export animation frames into")
       ("anim-max-iters-end", po::value<int>())
       ("anim-metric-arg-end", po::value<double>())
       ("anim-p-end", po::value<std::complex<double>>())
       ("anim-center-end", po::value<std::complex<double>>())
       ("anim-zoom-end", po::value<double>())
       ("anim-color-cutoff-end", po::value<double>())
       ("anim-path-start", po::value<std::complex<double>>())
       ("anim-path-end", po::value<std::complex<double>>())
       ("anim-line-a-start", po::value<std::complex<double>>())
       ("anim-line-a-end", po::value<std::complex<double>>())
       ("anim-line-b-start", po::value<std::complex<double>>())
       ("anim-line-b-end", po::value<std::complex<double>>())
       ("anim-circle-center-start", po::value<std::complex<double>>())
       ("anim-circle-center-end", po::value<std::complex<double>>())
       ("anim-circle-r-start", po::value<double>())
       ("anim-circle-r-end", po::value<double>())
       ("anim-shape-iters-start", po::value<int>()->default_value(0))
       ("anim-shape-iters-end", po::value<int>())
       ("anim-background,H", po::bool_switch()->default_value(false), "Creates animation without opening a window")
       ("cuda-path", po::value<std::string>())
       ("no-vsync", po::bool_switch()->default_value(false))
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
    opt.forceDisableIncrementalShapeTrans = vm["no-incremental-t"].as<bool>();
    opt.doublePrec = vm["double"].as<bool>();
    opt.p = vm["param"].as<std::complex<double>>();
    opt.viewportCenter = vm["center"].as<std::complex<double>>();
    opt.viewportBreadth = vm["zoom"].as<double>() / 2;
    opt.animBackground = vm["anim-background"].as<bool>();
    opt.enableVsync = !vm["no-vsync"].as<bool>();

    if(!vm["cuda-path"].empty()) {
        std::filesystem::path usrCudaPath = vm["cuda-path"].as<std::string>();
        if(std::filesystem::exists(usrCudaPath / "include"))
            opt.cudaIncludePath = usrCudaPath / "include";
        else
            throw std::runtime_error(std::string("The specified CUDA path ") + usrCudaPath.string() + " doesn't exist, or doesn't have an \"include\" subdirectory");
    } else {
        opt.cudaIncludePath = findCudaIncludePath();
    }

    if(vm.count("refs-path"))
        opt.refsPath = vm["refs-path"].as<std::string>();

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
        anim.totalFrames = vm["anim-length"].as<int>();
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
        addAnimParam(anim.shapeTransIteration, vm, "anim-shape-iters-start", "anim-shape-iters-end");

        if(vm.count("anim-line-a-start") && vm.count("anim-line-b-start")) {
            auto aEnd = vm.count("anim-line-a-end") ? vm["anim-line-a-end"].as<std::complex<double>>() : vm["anim-line-a-start"].as<std::complex<double>>();
            auto bEnd = vm.count("anim-line-b-end") ? vm["anim-line-b-end"].as<std::complex<double>>() : vm["anim-line-b-start"].as<std::complex<double>>();
            anim.shapeProps = {{
            {.shape = LINE,
                .line = {
                    .p1 = vm["anim-line-a-start"].as<std::complex<double>>(),
                    .p2 = vm["anim-line-b-start"].as<std::complex<double>>()
                }},
            {.shape = LINE,
                .line = {
                    .p1 = aEnd,
                    .p2 = bEnd
                }}
            }};
        } else if(vm.count("anim-circle-center-start") && vm.count("anim-circle-r-start")) {
            auto centerEnd = vm.count("anim-circle-center-end") ? vm["anim-circle-center-end"].as<std::complex<double>>() : vm["anim-circle-center-start"].as<std::complex<double>>();
            auto rEnd = vm.count("anim-circle-r-end") ? vm["anim-circle-r-end"].as<double>() : vm["anim-circle-r-start"].as<double>();
            anim.shapeProps = {{
            {.shape = CIRCLE,
                .circle = {
                    .center = vm["anim-circle-center-start"].as<std::complex<double>>(),
                    .r = vm["anim-circle-r-start"].as<double>()
                }},
            {.shape = CIRCLE,
                .circle = {
                     .center = centerEnd,
                     .r = rEnd
                }}
            }};
        }

        opt.animParams = std::move(anim);
    }

    return opt;
}