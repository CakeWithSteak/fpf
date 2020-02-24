#pragma once


#include "ExpressionNode.h"

class VariableNode : public ExpressionNode {
    char varName;
public:
    virtual std::optional<OperatorTraits> getOperator() const override;
    virtual std::string getCudaCode() const override;

    explicit VariableNode(char varName) : varName{varName} {}

    virtual NodeType type() const override;
};


