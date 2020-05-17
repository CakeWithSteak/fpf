#include "ConstantNode.h"

std::optional<OperatorTraits> ConstantNode::getOperator() const {
    return {};
}

std::string ConstantNode::getCudaCode() const {
    return "make_complex(" + std::to_string(num.real()) + "f," + std::to_string(num.imag()) + "f)";
}

NodeType ConstantNode::type() const {
    return NodeType::CONSTANT;
}

std::complex<float> ConstantNode::value() {
    return num;
}
