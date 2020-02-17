#pragma once
#include <string>
#include <stdexcept>
#include <list>

enum class TokenType {
    LEFT_PAREN,
    RIGHT_PAREN,
    COMMA,
    OPERATOR,
    NUMBER_LITERAL,
    VARIABLE
};

struct Token {
    TokenType type;
    std::string value;

    Token(TokenType type, const std::string& value) : type{type}, value{value} {}
    Token(TokenType type, char value) : type{type}, value{value} {}
};

using TokenList = std::list<Token>;