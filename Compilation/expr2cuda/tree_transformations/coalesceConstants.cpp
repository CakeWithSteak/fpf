#include "coalesceConstants.h"
#include "../expression_tree/ConstantNode.h"

ExpressionNode* traverse(ExpressionNode* node) {
    if(node->type() == NodeType::OPERATOR) {
        for(auto& child : node->children)
            child = traverse(child);

        //Merge neg(5) into -5
        if(node->getOperator()->name == "neg") {
            auto* child = node->children[0];
            if(child->type() == NodeType::CONSTANT) {
                auto* cons = dynamic_cast<ConstantNode*>(child);
                auto* merged = new ConstantNode(-cons->value());
                delete node;
                return merged;
            }
        } else if(node->getOperator()->name == "+" || node->getOperator()->name == "-") { //Merge {5,0} + {0, 3} into {5, 3}
            auto* child1 = node->children[0];
            auto* child2 = node->children[1];
            if(child1->type() == NodeType::CONSTANT && child2->type() == NodeType::CONSTANT) {
                auto* a = dynamic_cast<ConstantNode*>(child1);
                auto* b = dynamic_cast<ConstantNode*>(child2);
                ConstantNode* merged;
                if(node->getOperator()->name == "+")
                    merged = new ConstantNode(b->value() + a->value());
                else
                    merged = new ConstantNode(b->value() - a->value());
                delete node;
                return merged;
            }
        }
    }
    if(node->type() == NodeType::ROOT)
        node->children[0] = traverse(node->children[0]);
    return node;
}

void coalesceConstants(RootNode& tree) {
    traverse(&tree);
}