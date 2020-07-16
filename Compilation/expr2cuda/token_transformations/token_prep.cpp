#include <cassert>
#include "token_prep.h"
#include "../operators.h"
#include "../utils.h"

using std::next;
using std::prev;

TokenList::iterator findMatchingParen(TokenList::iterator it, TokenList::iterator end);
TokenList::iterator findTermEnd(TokenList::iterator it, TokenList::iterator end);

void unaryOpToFunction(TokenList& tl) {
    for(auto it = tl.begin(); it != tl.end(); ++it) {
        Token t = *it;
        if(mapContains(unaryOperatorLookup, t.value)) {
            if( it == tl.begin() ||
                prev(it)->type == TokenType::LEFT_PAREN ||
                prev(it)->type == TokenType::COMMA ||
                prev(it)->type == TokenType::OPERATOR)
            {
                const auto& replacementFunction = unaryOperatorLookup.at(t.value);
                if(replacementFunction.empty()) {
                    auto newIt = --it; // The iterator will be invalidated by erase
                    tl.erase(next(it));
                } else {
                    *it = Token(TokenType::OPERATOR, unaryOperatorLookup.at(t.value));
                    auto insertedParen = tl.insert(next(it), Token(TokenType::LEFT_PAREN, "("));
                    auto termEnd = findTermEnd(next(insertedParen), tl.end());
                    tl.insert(next(termEnd), Token(TokenType::RIGHT_PAREN, ")"));
                }
            }
        }
    }
}

void juxtaposeToExplicit(TokenList& tl) {
    const Token multiply = Token(TokenType::OPERATOR, '*');
    for(auto it = next(tl.begin()); it != tl.end(); ++it) {
        Token t = *it;
        if(t.type == TokenType::LEFT_PAREN && prev(it)->type == TokenType::RIGHT_PAREN)
            tl.insert(it, multiply);
        else if (prev(it)->type == TokenType::NUMBER_LITERAL || prev(it)->type == TokenType::VARIABLE) {
            if(t.type == TokenType::LEFT_PAREN ||
               t.type == TokenType::VARIABLE ||
               t.type == TokenType::NUMBER_LITERAL ||
               (t.type == TokenType::OPERATOR && getTraits(t).isFunction))
            {
                tl.insert(it, multiply);
            }
        }
    }
}

TokenList::iterator findMatchingParen(TokenList::iterator it, TokenList::iterator end) {
    assert(it->type == TokenType::LEFT_PAREN);
    assert(it != end);

    int parenLevel = 0;
    do {
        if(it->type == TokenType::LEFT_PAREN)
            ++parenLevel;
        else if(it->type == TokenType::RIGHT_PAREN)
            --parenLevel;
        ++it;
    } while(it != end && parenLevel > 0);

    if(parenLevel != 0)
        throw std::runtime_error("Unmatched parentheses in expression.");
    return --it;
}

//Finds the last token of a term
TokenList::iterator findTermEnd(TokenList::iterator it, TokenList::iterator end) {
    switch(it->type) {
        case TokenType::LEFT_PAREN:
            return findMatchingParen(it, end);
        case TokenType::VARIABLE:
        case TokenType::NUMBER_LITERAL:
            return it;
        case TokenType::OPERATOR: {
            auto temp = operatorLookup.find(it->value);
            if (temp != operatorLookup.end() && temp->second.isFunction) {
                return findMatchingParen(++it, end);
            } else {
                if(!mapContains(unaryOperatorLookup, it->value))
                    throw std::runtime_error("Syntax error at " + it->value);
                if(next(it) == end)
                    throw std::runtime_error("Syntax error: no operand found for " + it->value);
                return findTermEnd(++it, end);
            }
        }
        default:
            throw std::runtime_error("Syntax error at " + it->value);
    }
}