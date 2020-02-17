#pragma once
#include <vector>
#include <optional>
#include "../OperatorTraits.h"

class ExpressionNode {
public:
    std::vector<ExpressionNode*> children;
    [[nodiscard]] virtual std::optional<OperatorTraits> getOperator() const = 0;
    [[nodiscard]] virtual std::string getCudaCode() const = 0;
    virtual ~ExpressionNode();
};