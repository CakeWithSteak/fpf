#include "compileExpression.h"
#include "expr2cuda/token_transformations/tokenize.h"
#include "expr2cuda/token_transformations/token_prep.h"
#include "expr2cuda/token_transformations/infixtoprefix.h"
#include "expr2cuda/tree_transformations/buildTree.h"
#include "expr2cuda/tree_transformations/coalesceConstants.h"

std::string compileExpression(std::string_view sv) {
    auto tokens = tokenize(sv);
    unaryOpToFunction(tokens);
    juxtaposeToExplicit(tokens);
    auto prefix = infixToPrefix(tokens);
    auto tree = buildTree(prefix);
    coalesceConstants(tree);
    return tree.getCudaCode();
}