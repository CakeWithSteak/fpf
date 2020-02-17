#include "tokenize.h"
#include "operators.h"
#include "../utils.h"
#include <cctype>

Token createNumberLiteral(std::string_view sv, int& i);
bool isOperatorChar(char c);

TokenList tokenize(std::string_view sv) {
    TokenList res;

    for(int i = 0; i < sv.size(); ++i) {
        char c = sv[i];

        if(c == '(')
            res.emplace_back(TokenType::LEFT_PAREN, c);
        else if(c == ')')
            res.emplace_back(TokenType::RIGHT_PAREN, c);
        else if(c == ',')
            res.emplace_back(TokenType::COMMA, c);
        else if(std::isdigit(c) || c == '.' || c == 'i')
            res.push_back(createNumberLiteral(sv, i));
        else if(!std::isspace(c)) {
            std::string temp;
            int foundOperatorAt = -1;
            for(int k = i; k < sv.size() && isOperatorChar(sv[k]); ++k) {
                temp.push_back(sv[k]);
                if(mapContains(operatorLookup, temp) || mapContains(unaryOperatorLookup, temp)) {
                    res.emplace_back(TokenType::OPERATOR, temp);
                    foundOperatorAt = k;
                    break;
                }
            }
            if(foundOperatorAt != -1)
                i = foundOperatorAt;
            else
                res.emplace_back(TokenType::VARIABLE, c);
        }
    }
    return res;
}

Token createNumberLiteral(std::string_view sv, int& i) {
    std::string val;
    bool hadDecimalPoint = false;
    char c = sv[i];
    do {
        if(c == '.') {
            if(hadDecimalPoint)
                throw std::runtime_error("Failed to parse exception: Multiple decimal points in one number");
            hadDecimalPoint = true;
        }

        val.push_back(c);
        ++i;
        if(c == 'i')
            break;
        c = sv[i];
    } while(std::isdigit(c) || c == '.' || c == 'i');
    --i;
    return {TokenType::NUMBER_LITERAL, val};
}

bool isOperatorChar(char c) {
    const std::string nonOperatorChars = "(),.";
    return !(std::isspace(c) || std::isdigit(c) || nonOperatorChars.find(c) != std::string::npos);
}