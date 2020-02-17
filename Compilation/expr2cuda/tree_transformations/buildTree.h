#pragma once

#include "../token_transformations/Token.h"
#include "../expression_tree/RootNode.h"

RootNode buildTree(const TokenList& tl);