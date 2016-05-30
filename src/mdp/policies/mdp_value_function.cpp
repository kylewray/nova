/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2015 Kyle Hollins Wray, University of Massachusetts
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


#include "policies/mdp_value_function.h"
#include "error_codes.h"
#include "constants.h"

#include <stdio.h>

namespace nova {

int mdp_value_function_uninitialize(MDPValueFunction *policy)
{
    if (policy == nullptr) {
        fprintf(stderr, "Error[mdp_value_function_uninitialize]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    policy->n = 0;
    policy->m = 0;
    policy->r = 0;

    if (policy->S != nullptr) {
        delete [] policy->S;
    }
    policy->S = nullptr;

    if (policy->V != nullptr) {
        delete [] policy->V;
    }
    policy->V = nullptr;

    if (policy->pi != nullptr) {
        delete [] policy->pi;
    }
    policy->pi = nullptr;

    //delete policy;
    //policy = nullptr;

    return NOVA_SUCCESS;
}

}; // namespace nova

