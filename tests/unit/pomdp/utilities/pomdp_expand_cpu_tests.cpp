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


#include "utilities/pomdp_expand_cpu.h"

#include <gtest/gtest.h>

#include "utilities/pomdp_model_cpu.h"
#include "error_codes.h"
#include "constants.h"

#include "unit/pomdp/pomdp_tests.h"

namespace nova {
namespace tests {

TEST(POMDPExpandCPU, random)
{
    nova::POMDP *pomdp = create_two_state_pomdp();

    unsigned int maxNonZeroValues = 0;
    float *Bnew = nullptr;
    int result = nova::pomdp_expand_random_cpu(pomdp, 42, maxNonZeroValues, Bnew);

    result = nova::pomdp_uninitialize_cpu(pomdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete pomdp;
}

TEST(POMDPExpandCPU, badRandom)
{

}

}; // namespace tests
}; // namespace nova

