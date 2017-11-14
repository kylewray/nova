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


#include <nova/mdp/utilities/ssp_stack.h>

#include <nova/mdp/mdp.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

#include <stdio.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <math.h>

#include <iostream>

namespace nova {

int ssp_stack_create(SSPStack &stack)
{
    if (stack.stack != nullptr || stack.maxStackSize == 0) {
        fprintf(stderr, "Error[ssp_stack_create]: %s\n", "Invalid data.");
        return NOVA_ERROR_INVALID_DATA;
    }

    stack.stack = new unsigned int[stack.maxStackSize];
    stack.stackSize = 0;

    return NOVA_SUCCESS;
}


int ssp_stack_pop(SSPStack &stack, unsigned int &s)
{
    if (stack.stack == nullptr || stack.stackSize == 0) {
        fprintf(stderr, "Error[ssp_stack_pop]: %s\n", "Stack is empty.");
        return NOVA_ERROR_EMPTY_CONTAINER;
    }

    s = stack.stack[stack.stackSize - 1];
    stack.stackSize--;

    return NOVA_SUCCESS;
}


int ssp_stack_push(SSPStack &stack, unsigned int s)
{
    if (stack.stack == nullptr || stack.stackSize >= stack.maxStackSize) {
        fprintf(stderr, "Error[ssp_stack_push]: %s\n",
                "Stack is full. Out of memory reserved for it.");
        return NOVA_ERROR_OUT_OF_MEMORY;
    }

    stack.stack[stack.stackSize] = s;
    stack.stackSize++;

    return NOVA_SUCCESS;
}


bool ssp_stack_in(SSPStack &stack, unsigned int s)
{
    if (stack.stack == nullptr) {
        fprintf(stderr, "Warning[ssp_stack_in]: %s\n",
                "Stack has not yet been created. Trivially, the element is not in the stack.");
    }

    for (unsigned int i = 0; i < stack.stackSize; i++) {
        if (stack.stack[i] == s) {
            return true;
        }
    }

    return false;
}


int ssp_stack_destroy(SSPStack &stack)
{
    if (stack.stack == nullptr) {
        fprintf(stderr, "Error[ssp_stack_destroy]: %s\n",
                "Stack has not yet been created. There is nothing to destroy.");
        return NOVA_ERROR_INVALID_DATA;
    }

    delete [] stack.stack;
    stack.stack = nullptr;
    stack.stackSize = 0;

    return NOVA_SUCCESS;
}

}; // namespace nova

