/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2014 Kyle Wray
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


#include "../include/nova_cuda.h"

#include <math.h>

// This is not C++0x, unfortunately.
#define nullptr NULL

__global__ void bellman_update(unsigned int n, unsigned int m, float *T, float *R,
		float gamma, float *V, float *VPrime, unsigned int *pi)
{
	// The current state as a function of the blocks and threads.
	int s;

	// The intermediate Q(s, a) value.
	float Qsa;

	// The 1-d index version of the 3-d arrays in the innermost loop.
	int k;

	// Compute the index of the state. Return if it is beyond the state.
	s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= n) {
		return;
	}

	// Compute max_{a in A} Q(s, a).
	for (int a = 0; a < m; a++) {
		// Compute Q(s, a) for this action.
		Qsa = 0.0f;
		for (int sp = 0; sp < n; sp++) {
			k = s * n * m + a * n + sp;
			Qsa += T[k] * (R[k] + gamma * V[sp]);
		}

		if (a == 0 || Qsa > VPrime[s]) {
			VPrime[s] = Qsa;
			pi[s] = a;
		}
	}
}

int value_iteration(unsigned int n, unsigned int m, const float ***T, const float ***R,
		float Rmax, float gamma, float epsilon, float *V, unsigned int *pi,
		unsigned int numBlocks, unsigned int numThreads)
{
	// The device pointers for the MDP: T and R.
	float *d_T;
	float *d_R;

	// The host and device pointers for the value functions: V and VPrime.
	float *d_V;
	float *d_VPrime;

	// The device pointer for the final policy: pi.
	unsigned int *d_pi;

	// First, ensure data is valid.
	if (n == 0 || m == 0 || T == nullptr || R == nullptr ||
			gamma < 0.0f || gamma >= 1.0f || pi == nullptr) {
		return -1;
	}

	// Also ensure that there are enough blocks and threads to run the program.
	if (numBlocks * numThreads < n) {
		return -2;
	}

	// Next, determine how many iterations it will have to run.
	int iterations = (int)std::ceil(std::log(2.0f * Rmax / (epsilon * (1.0 - gamma)) / std::log(1.0 / gamma)));

	// Allocate the device-side memory.
	cudaMalloc(&d_T, n * m * n * sizeof(float));
	cudaMalloc(&d_R, n * m * n * sizeof(float));

	cudaMalloc(&d_V, n * sizeof(float));
	cudaMalloc(&d_VPrime, n * sizeof(float));

	cudaMalloc(&d_pi, n * sizeof(unsigned int));

	// Ensure that V and pi are initialized to zero to start.
	for (int s = 0; s < n; s++) {
		V[s] = 0.0f;
		pi[s] = 0;
	}

	// Copy the data from host to device.
	cudaMemcpy(d_T, T, n * m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, R, n * m * n * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_V, V, n * m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_VPrime, V, n * m * n * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_pi, pi, n * m * n * sizeof(float), cudaMemcpyHostToDevice);

	//	for (int s = 0; s < n; s++) {
	//		for (int a = 0; a < m; a++) {
	//			for (int sp = 0; sp < n; sp++) {
	//				d_T[s * n * m + a * n + sp] = T[s][a][sp];
	//				d_R[s * n * m + a * n + sp] = R[s][a][sp];
	//			}
	//		}
	//	}

	// Execute value iteration for these number of iterations. For each iteration, however,
	// we will run the state updates in parallel.
	for (int i = 0; i < iterations; i++) {
		if (i % 2 == 0) {
			bellman_update<<< numBlocks, numThreads >>>(n, m, d_T, d_R, gamma, d_V, d_VPrime, d_pi);
		} else {
			bellman_update<<< numBlocks, numThreads >>>(n, m, d_T, d_R, gamma, d_VPrime, d_V, d_pi);
		}
	}

	// Copy the final result, both V and pi, from device to host.
	if (iterations % 2 == 0) {
		cudaMemcpy(V, d_V, n * m * n * sizeof(float), cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(V, d_VPrime, n * m * n * sizeof(float), cudaMemcpyDeviceToHost);
	}
	cudaMemcpy(pi, d_pi, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// Free the device-side memory.
	cudaFree(d_T);
	cudaFree(d_R);

	cudaFree(d_V);
	cudaFree(d_VPrime);

	cudaFree(d_pi);

	return 0;
}
