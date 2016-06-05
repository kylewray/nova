/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2016 Kyle Hollins Wray, University of Massachusetts
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

#include <gtest/gtest.h>

#include "error_codes.h"
#include "constants.h"


namespace nova {
namespace tests {

TEST(MDPValueFunction, uninitialization)
{
    nova::MDPValueFunction *mdpValueFunction = new nova::MDPValueFunction();
    mdpValueFunction->n = 2;
    mdpValueFunction->m = 2;
    mdpValueFunction->r = 2;

    mdpValueFunction->S = new unsigned int[2];
    mdpValueFunction->V = new float[2];
    mdpValueFunction->pi = new unsigned int[2];

    int result = nova::mdp_value_function_uninitialize(mdpValueFunction);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_EQ(mdpValueFunction->S, nullptr);
    if (mdpValueFunction->S != nullptr) {
        delete [] mdpValueFunction->S;
    }

    EXPECT_EQ(mdpValueFunction->V, nullptr);
    if (mdpValueFunction->V != nullptr) {
        delete [] mdpValueFunction->V;
    }

    EXPECT_EQ(mdpValueFunction->pi, nullptr);
    if (mdpValueFunction->pi != nullptr) {
        delete [] mdpValueFunction->pi;
    }

    delete mdpValueFunction;
}

TEST(MDPValueFunction, badUninitializations)
{
    nova::MDPValueFunction *mdpValueFunction = nullptr;
    int result = nova::mdp_value_function_uninitialize(mdpValueFunction);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
}

}; // namespace tests
}; // namespace nova

