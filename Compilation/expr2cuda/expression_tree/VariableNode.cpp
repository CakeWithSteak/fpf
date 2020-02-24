#include "VariableNode.h"

std::optional<OperatorTraits> VariableNode::getOperator() const {
    return {};
}

std::string VariableNode::getCudaCode() const {
    return std::string{varName};
}

NodeType VariableNode::type() const{
    return NodeType::VARIABLE;
}