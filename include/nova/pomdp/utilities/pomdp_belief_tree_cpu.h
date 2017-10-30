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


#ifndef NOVA_POMDP_BELIEF_TREE_CPU_H
#define NOVA_POMDP_BELIEF_TREE_CPU_H


namespace nova {

#define NOVA_BELIEF_TREE_BIN_SIZE 10

/**
 *  A belief tree object for compactly storing beliefs approximately.
 */
typedef struct NovaBeliefTree {
    unsigned int state;
    NovaBeliefTree *successors[NOVA_BELIEF_TREE_BIN_SIZE];
} BeliefTree;

/**
 *  
 */
extern "C" void pomdp_belief_tree_initialize_cpu(BeliefTree *tree, unsigned int state);

/**
 *  
 */
void pomdp_belief_tree_uninitialize_cpu(BeliefTree *tree, unsigned int n);

/**
 *  
 */
bool pomdp_belief_tree_has_belief_cpu(BeliefTree *tree, unsigned int n, float *b);

/**
 *  
 */
void pomdp_belief_tree_insert_belief_cpu(BeliefTree *tree, unsigned int n, float *b);

};


#endif // NOVA_POMDP_BELIEF_TREE_CPU_H


