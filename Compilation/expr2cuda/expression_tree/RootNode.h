#pragma once
#include "ExpressionNode.h"

class RootNode : public ExpressionNode {
public:
    virtual std::optional<OperatorTraits> getOperator() const override;

    virtual std::string getCudaCode() const override;

    RootNode() = default;
    RootNode(const RootNode& other) = delete;
    RootNode(RootNode&& other) noexcept;
    RootNode& operator= (RootNode&& other) noexcept;
    RootNode& operator= (RootNode& other) = delete;

    virtual NodeType type() const override;
};


