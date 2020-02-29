#pragma once
#include "../expression_tree/RootNode.h"

//Merges constants (eg. {5, 0} + {0, 3} -> {5, 3}
void coalesceConstants(RootNode& tree);