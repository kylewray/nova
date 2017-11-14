/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2017 Kyle Hollins Wray, University of Massachusetts
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *  the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 *  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 *  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 *  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#include <nova/pomdp/utilities/pomdp_belief_tree.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

#include <stdio.h>
#include <cstdlib>
#include <cmath>

namespace nova {

void pomdp_belief_tree_initialize(BeliefTree *tree, unsigned int state)
{
    tree->state = state;
    for (unsigned int i = 0; i < NOVA_BELIEF_TREE_BIN_SIZE; i++) {
        tree->successors[i] = nullptr;
    }
}


bool pomdp_belief_tree_has_belief(BeliefTree *tree, unsigned int n, float *b)
{
    // If we get to the end, then we have found a path, and therefore the belief exists in the tree.
    if (tree->state >= n) {
        return true;
    }

    // We compute the index of the belief value for this state. Then check if the successor is defined.
    unsigned int index = (unsigned int)roundf(b[tree->state] * (float)(NOVA_BELIEF_TREE_BIN_SIZE - 1));

    // If it not defined, then there is no path and thus the belief is not in the tree. Otherwise, recurse.
    if (tree->successors[index] == nullptr) {
        return false;
    } else {
        return pomdp_belief_tree_has_belief(tree->successors[index], n, b);
    }
}


void pomdp_belief_tree_insert_belief(BeliefTree *tree, unsigned int n, float *b)
{
    // Note: We assume that we already know the belief does not exist in the tree!

    // If we get to the end, then we have found a path, and therefore the belief was successfully created.
    if (tree->state >= n) {
        return;
    }

    // We compute the index of the belief value for this state, initialize the successor if needed,
    // and continue to recurse.
    unsigned int index = (unsigned int)roundf(b[tree->state] * (float)(NOVA_BELIEF_TREE_BIN_SIZE - 1));

    if (tree->successors[index] == nullptr) {
        tree->successors[index] = new BeliefTree();
        pomdp_belief_tree_initialize(tree->successors[index], tree->state + 1);
    }

    pomdp_belief_tree_insert_belief(tree->successors[index], n, b);
}


void pomdp_belief_tree_uninitialize(BeliefTree *tree, unsigned int n)
{
    if (tree->state >= n) {
        return;
    }

    for (unsigned int i = 0; i < NOVA_BELIEF_TREE_BIN_SIZE; i++) {
        if (tree->successors[i] != nullptr) {
            pomdp_belief_tree_uninitialize(tree->successors[i], n);
            delete tree->successors[i];
            tree->successors[i] = nullptr;
        }
    }
}

}; // namespace nova

