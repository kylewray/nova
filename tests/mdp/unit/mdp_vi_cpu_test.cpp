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


#include "algorithms/mdp_vi_cpu.h"

#include <gtest/gtest.h>

#include "error_codes.h"
#include "constants.h"


TEST(MDPVICPU, handlesInitialization)
{
    nova::MDP mdp;
    mdp.n = 2;

    nova::MDPVICPU vi;
    vi.Vinitial = new float[2];
    vi.Vinitial[0] = -1.0f;
    vi.Vinitial[1] = 1.0f;

    int result = 0;

    result = nova::mdp_vi_initialize_cpu(&mdp, &vi);

    ASSERT_EQ(result, NOVA_SUCCESS);
    EXPECT_EQ(vi.currentHorizon, 0);

    ASSERT_NE(vi.V, nullptr);
    EXPECT_EQ(vi.V[0], -1.0f);
    EXPECT_EQ(vi.V[1], 1.0f);

    ASSERT_NE(vi.Vprime, nullptr);
    EXPECT_EQ(vi.Vprime[0], -1.0f);
    EXPECT_EQ(vi.Vprime[1], 1.0f);

    ASSERT_NE(vi.pi, nullptr);
    EXPECT_EQ(vi.pi[0], 0);
    EXPECT_EQ(vi.pi[1], 0);

    result = nova::mdp_vi_uninitialize_cpu(&mdp, &vi);

    ASSERT_EQ(result, NOVA_SUCCESS);

    EXPECT_EQ(vi.currentHorizon, 0);
    EXPECT_EQ(vi.V, nullptr);
    EXPECT_EQ(vi.Vprime, nullptr);
    EXPECT_EQ(vi.pi, nullptr);

    delete [] vi.Vinitial;
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

