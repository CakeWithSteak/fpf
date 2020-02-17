#include "OperatorNode.h"

std::optional<OperatorTraits> OperatorNode::getOperator() const {
    return opTraits;
}

std::string OperatorNode::getCudaCode() const {
    std::string result(opTraits.cudaName + "(");
    for(int i = opTraits.arity - 1; i >= 0; --i) { // Prefix conversion swaps the order of operands, so swap it back here
        result += children[i]->getCudaCode() + ",";
    }
    result[result.size() - 1] = ')'; // Cut the trailing comma
    return result;
}
