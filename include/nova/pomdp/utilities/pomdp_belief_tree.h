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


#ifndef NOVA_POMDP_BELIEF_TREE_H
#define NOVA_POMDP_BELIEF_TREE_H


namespace nova {

#define NOVA_BELIEF_TREE_BIN_SIZE 10

/**
 *  A belief tree object for compactly storing beliefs approximately.
 *
 *  The main idea is to store beliefs in an approximate tree instead of a big matrix.
 *  The tree has a height of the number of states (n) and each branch represents
 *  a particular probability bin. This lets us check if a belief basically exists
 *  in O(n) time, rather than O(rn) where r is the number of beliefs (very large).
 *
 *  @param  state       The state of this node in the tree.
 *  @param  successors  An approximate form of belief using bins.
 */
typedef struct NovaBeliefTree {
    unsigned int state;
    NovaBeliefTree *successors[NOVA_BELIEF_TREE_BIN_SIZE];
} BeliefTree;

/**
 *  Initalize the belief tree by assigning the starting root state.
 *  @param  tree    The belief tree to initialize.
 *  @param  state   The starting root state of the tree.
 */
extern "C" void pomdp_belief_tree_initialize(BeliefTree *tree, unsigned int state);

/**
 *  Check if the tree has a particular belief.
 *  @param  tree    The belief tree to check.
 *  @param  n       The number of states in the belief.
 *  @param  b       The belief as an n-array.
 *  @return True if the belief is in the tree; false otherwise.
 */
extern "C" bool pomdp_belief_tree_has_belief(BeliefTree *tree, unsigned int n, float *b);

/**
 *  Insert the belief into the belief tree with default state value of n.
 *  @param  tree    The belief tree that will insert the belief.
 *  @param  n       The number of states in the belief.
 *  @param  b       The belief as an n-array.
 */
extern "C" void pomdp_belief_tree_insert_belief(BeliefTree *tree, unsigned int n, float *b);

/**
 *  Uninitialize the belief tree. Note: Internally it uses n to determine leaf nodes.
 *  @param  tree    The belief tree to free.
 *  @param  n       The number of states in the beliefs inserted into the tree.
 */
extern "C" void pomdp_belief_tree_uninitialize(BeliefTree *tree, unsigned int n);

};


#endif // NOVA_POMDP_BELIEF_TREE_H


