#include <string>
#include <stack>
#include "buildTree.h"
#include "../utils.h"
#include "../expression_tree/RootNode.h"
#include "../expression_tree/OperatorNode.h"
#include "../expression_tree/VariableNode.h"
#include "../expression_tree/ConstantNode.h"

std::complex<float> interpretLiteral(std::string_view sv);

RootNode buildTree(const TokenList& tl) {
    RootNode root;
    std::stack<std::pair<ExpressionNode*, int>> nodeStack; //Stores node and arity
    nodeStack.emplace(&root, 1);

    for(const Token& t : tl) {
        if(nodeStack.empty())
            throw std::runtime_error("Parse error: Invalid number of operands for a function");
        switch(t.type) {
            case TokenType::OPERATOR: {
                auto traits = getTraits(t);
                auto* node = new OperatorNode(traits);
                nodeStack.top().first->children.push_back(node);
                nodeStack.emplace(node, traits.arity);
                break;
            }
            case TokenType::VARIABLE:
                nodeStack.top().first->children.push_back(new VariableNode(t.value[0]));
                break;
            case TokenType::NUMBER_LITERAL:
                nodeStack.top().first->children.push_back(new ConstantNode(interpretLiteral(t.value)));
                break;
            default:
                throw std::runtime_error("Parse error: invalid token during tree creation");
        }
        while(!nodeStack.empty() && nodeStack.top().second == nodeStack.top().first->children.size())
            nodeStack.pop();
    }
    return root;
}

std::complex<float> interpretLiteral(std::string_view sv) {
    float real = 0, imag = 0;
    if(sv.back() == 'i') {
        imag = (sv.size() > 1) ? std::stof(std::string(sv.substr(0, sv.size() - 1))) : 1;
    } else {
        real = std::stof(std::string(sv));
    }
    return {real, imag};
}