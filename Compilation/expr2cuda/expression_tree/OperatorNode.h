#pragma once

#include <utility>

#include "ExpressionNode.h"

class OperatorNode : public ExpressionNode {
    OperatorTraits opTraits;
public:
    virtual std::optional<OperatorTraits> getOperator() const override;
    virtual std::string getCudaCode() const override;

    explicit OperatorNode(OperatorTraits traits) : opTraits{traits} {}
};


