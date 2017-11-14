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


#ifndef NOVA_SSP_STACK_H
#define NOVA_SSP_STACK_H

namespace nova {

/**
 *  The SSP stack structure for SSPs.
 *  @param  maxStackSize    The maximum stack size to allocate and check for a stack.
 *  @param  stackSize       The current stack size.
 *  @param  stack           A pointer to the actual stack data in memory.
 */
typedef struct NovaSSPStack {
    unsigned int maxStackSize;
    unsigned int stackSize;
    unsigned int *stack;
} SSPStack;

/**
 *  Create a stack, reserving memory, given the max stack size in the stack object.
 *  @param  stack   The stack object.
 *  @return NOVA_SUCCESS if successful; NOVA_ERROR_INVALID_DATA if the stack was already created.
 */
int ssp_stack_create(SSPStack &stack);

/**
 *  Pop an element off of the stack.
 *  @param  stack   The stack object.
 *  @param  s       The resultant element when popped off the stack.
 *  @return NOVA_SUCCESS if successful; NOVA_ERROR_EMPTY_CONTAINER if the stack is empty.
 */
int ssp_stack_pop(SSPStack &stack, unsigned int &s);

/**
 *  Push an element onto the stack.
 *  @param  stack   The stack object.
 *  @param  s       The element to push on the stack.
 *  @return NOVA_SUCCESS if successful; NOVA_ERROR_OUT_OF_MEMORY if the stack is full.
 */
int ssp_stack_push(SSPStack &stack, unsigned int s);

/**
 *  Check if the element s is in the stack.
 *  @param  stack   The stack object.
 *  @param  s       The element to look for in the stack.
 */
bool ssp_stack_in(SSPStack &stack, unsigned int s);

/**
 *  Destroy (free) the memory in the stack.
 *  @param  stack   The stack object.
 *  @return NOVA_SUCCESS if successful; NOVA_ERROR_INVALID_DATA if the stack was not yet created.
 */
int ssp_stack_destroy(SSPStack &stack);

}; // namespace nova

#endif // NOVA_SSP_STACK_H

