#include <algorithm>
#include "ExpressionNode.h"

ExpressionNode::~ExpressionNode() {
    std::for_each(children.begin(), children.end(), [](ExpressionNode* child){delete child;});
}