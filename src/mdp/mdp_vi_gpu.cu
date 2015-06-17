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


#include "mdp_vi_gpu.h"
#include "error_codes.h"

#include <cmath>
#include <stdio.h>


// This is determined by hardware, so what is below is a 'safe' guess. If this is
// off, the program might return 'nan' or 'inf'.
#define FLT_MAX 1e+35


__global__ void mdp_bellman_update_gpu(unsigned int n, unsigned int ns, unsigned int m, float gamma,
        const int *S, const float *T, const float *R, const float *V, float *VPrime, unsigned int *pi)
{
    // The current state as a function of the blocks and threads.
    int s;

    // The intermediate Q(s, a) value.
    float Qsa;

    // The index within S and T (i.e., in n*s*ns).
    int index;

    // The true successor state index (in 0 to n-1), resolved using S.
    int spindex;

    // Compute the index of the state. Return if it is beyond the state.
    s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) {
        return;
    }

    // Nvidia GPUs follow IEEE floating point standards, so this should be safe.
    VPrime[s] = -FLT_MAX;

    // Compute max_{a in A} Q(s, a).
    for (int a = 0; a < m; a++) {
        // Compute Q(s, a) for this action.
        Qsa = R[s * m + a];

        for (int sp = 0; sp < ns; sp++) {
            index = s * m * ns + a * ns + sp;

            spindex = S[index];
            if (spindex < 0) {
                break;
            }

            Qsa += gamma * T[index] * V[spindex];
        }

        __syncthreads();

        if (a == 0 || Qsa > VPrime[s]) {
            VPrime[s] = Qsa;
            pi[s] = a;
        }

        __syncthreads();
    }
}


int mdp_vi_complete_gpu(MDP *mdp, unsigned int numThreads,
                float *V, unsigned int *pi)
{
    // The host and device pointers for the value functions: V and VPrime.
    float *d_V;
    float *d_VPrime;

    // The device pointer for the final policy: pi.
    unsigned int *d_pi;

    // The number of blocks
    unsigned int numBlocks;

    // The final return message code, which is set during freeing memory at the end.
    unsigned int result;

    // First, ensure data is valid.
    if (mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->gamma < 0.0f || mdp->gamma >= 1.0f || mdp->horizon < 1 ||
            V == nullptr || pi == nullptr) {
        return NOVA_ERROR_INVALID_DATA;
    }

    // Ensure threads are correct.
    if (numThreads % 32 != 0) {
        fprintf(stderr, "Error[value_iteration]: %s", "Invalid number of threads.");
        return NOVA_ERROR_INVALID_CUDA_PARAM;
    }

    numBlocks = (unsigned int)((float)mdp->n / (float)numThreads) + 1;

    // Allocate the device-side memory.
    if (cudaMalloc(&mdp->d_S, mdp->n * mdp->m * mdp->ns * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the successor states.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMalloc(&mdp->d_T, mdp->n * mdp->m * mdp->ns * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the state transitions.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMalloc(&mdp->d_R, mdp->n * mdp->m * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the rewards.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    if (cudaMalloc(&d_V, mdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the value function.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMalloc(&d_VPrime, mdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the value function (prime).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    if (cudaMalloc(&d_pi, mdp->n * sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the policy (pi).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from host to device.
    if (cudaMemcpy(mdp->d_S, mdp->S, mdp->n * mdp->m * mdp->ns * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the successors.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }
    if (cudaMemcpy(mdp->d_T, mdp->T, mdp->n * mdp->m * mdp->ns * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the state transitions.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }
    if (cudaMemcpy(mdp->d_R, mdp->R, mdp->n * mdp->m * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the rewards.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    if (cudaMemcpy(d_V, V, mdp->n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the value function.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }
    if (cudaMemcpy(d_VPrime, V, mdp->n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the value function (prime).");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    if (cudaMemcpy(d_pi, pi, mdp->n * sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the policy (pi).");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    // Execute value iteration for these number of iterations. For each iteration, however,
    // we will run the state updates in parallel.
    for (int i = 0; i < mdp->horizon; i++) {
        printf("Iteration %d / %d -- GPU Version\n", i, mdp->horizon);

        if (i % 2 == 0) {
            mdp_bellman_update_gpu<<< numBlocks, numThreads >>>(mdp->n, mdp->ns, mdp->m, mdp->gamma, mdp->d_S, mdp->d_T, mdp->d_R, d_V, d_VPrime, d_pi);
        } else {                                                                                     
            mdp_bellman_update_gpu<<< numBlocks, numThreads >>>(mdp->n, mdp->ns, mdp->m, mdp->gamma, mdp->d_S, mdp->d_T, mdp->d_R, d_VPrime, d_V, d_pi);
        }

        // Check if there was an error executing the kernel.
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "Error[value_iteration]: %s",
                            "Failed to execute the 'initialization of alphaBA' kernel.");
            return NOVA_ERROR_KERNEL_EXECUTION;
        }

        // Wait for the kernel to finish before looping more.
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "Error[value_iteration]: %s",
                        "Failed to synchronize the device after 'initialization of alphaBA' kernel.");
            return NOVA_ERROR_DEVICE_SYNCHRONIZE;
        }
    }

    // Copy the final result, both V and pi, from device to host.
    if (mdp->horizon % 2 == 1) {
        if (cudaMemcpy(V, d_V, mdp->n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[value_iteration]: %s",
                    "Failed to copy memory from device to host for the value function.");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    } else {
        if (cudaMemcpy(V, d_VPrime, mdp->n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[value_iteration]: %s",
                    "Failed to copy memory from device to host for the value function (prime).");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    }
    if (cudaMemcpy(pi, d_pi, mdp->n * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from device to host for the policy (pi).");
        return NOVA_ERROR_MEMCPY_TO_HOST;
    }

    // Free the device-side memory. Note that we continue on error, since it is just cleaning
    // memory, but it is always nice to know if this failed anywhere.
    result = NOVA_SUCCESS;

    if (cudaFree(mdp->d_S) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the successor states.");
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (cudaFree(mdp->d_T) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the state transitions.");
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (cudaFree(mdp->d_R) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the rewards.");
        result = NOVA_ERROR_DEVICE_FREE;
    }

    if (cudaFree(d_V) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the value function.");
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (cudaFree(d_VPrime) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the value function (prime).");
        result = NOVA_ERROR_DEVICE_FREE;
    }

    if (cudaFree(d_pi) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the policy (pi).");
        result = NOVA_ERROR_DEVICE_FREE;
    }

    return result;
}

