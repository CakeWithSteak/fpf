#include "RootNode.h"

std::optional<OperatorTraits> RootNode::getOperator() const {
    return {};
}

std::string RootNode::getCudaCode() const {
    if(children.empty())
        throw std::runtime_error("The RootNode you tried to access is empty.");
    return "return " + children[0]->getCudaCode() + ";";
}

RootNode::RootNode(RootNode&& other) noexcept {
    if(!other.children.empty()) {
        children.push_back(other.children[0]);
        other.children.pop_back();
    }
}

RootNode& RootNode::operator=(RootNode&& other) noexcept {
    if(other.children.empty())
        return *this;

    if(!children.empty()) {
        delete children[0];
        children[0] = other.children[0];
    } else {
        children.push_back(other.children[0]);
    }

    other.children.pop_back();
    return *this;
}

NodeType RootNode::type() const{
    return NodeType::ROOT;
}