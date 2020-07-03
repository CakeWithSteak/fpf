#include <stack>
#include "infixtoprefix.h"
#include "../OperatorTraits.h"
#include "../utils.h"

TokenList infixToPrefix(const TokenList& tl) {
    std::stack<Token> operatorStack;
    TokenList output;

    for(const Token& t : tl) {
        switch(t.type) {
            case TokenType::NUMBER_LITERAL:
            case TokenType::VARIABLE:
                output.push_front(t);
                break;
            case TokenType::OPERATOR: {
                auto traits = getTraits(t);
                if(traits.isFunction) {
                    operatorStack.push(t);
                } else {
                    while(!operatorStack.empty() && operatorStack.top().type == TokenType::OPERATOR) {
                        Token t2 = operatorStack.top();
                        auto traits2 = getTraits(t2);
                        if((traits.associativity == LEFT_ASSOCIATIVE && traits.precedence <= traits2.precedence) ||
                           (traits.associativity == RIGHT_ASSOCIATIVE && traits.precedence < traits2.precedence) ||
                            traits2.isFunction)
                        {
                            output.push_front(t2);
                            operatorStack.pop();
                        } else break;
                    }
                    operatorStack.push(t);
                }
            }
            break;
            case TokenType::COMMA:
                while(!operatorStack.empty() && operatorStack.top().type != TokenType::LEFT_PAREN) {
                    output.push_front(operatorStack.top());
                    operatorStack.pop();
                }
                break;
            case TokenType::LEFT_PAREN:
                operatorStack.push(t);
                break;
            case TokenType::RIGHT_PAREN:
                while(!operatorStack.empty() && operatorStack.top().type != TokenType::LEFT_PAREN) {
                    output.push_front(operatorStack.top());
                    operatorStack.pop();
                }
                if(operatorStack.empty())
                    throw std::runtime_error("Mismatched parentheses in expression.");
                operatorStack.pop(); // Pop the left paren
                break;
        }
    }
    while(!operatorStack.empty()) {
        Token t = operatorStack.top();
        if(t.type == TokenType::LEFT_PAREN)
            throw std::runtime_error("Mismatched parentheses in expression.");
        output.push_front(t);
        operatorStack.pop();
    }
    return output;
}

