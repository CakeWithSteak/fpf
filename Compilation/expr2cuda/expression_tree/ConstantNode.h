#pragma once


#include <complex>
#include "ExpressionNode.h"

class ConstantNode : public ExpressionNode {
    std::complex<float> num;
public:
    virtual std::optional<OperatorTraits> getOperator() const override;
    virtual std::string getCudaCode() const override;

    explicit ConstantNode(const std::complex<float>& num) : num(num) {}
};


