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


#include <nova/pomdp/algorithms/pomdp_pbvi_gpu.h>
#include <nova/pomdp/utilities/pomdp_model_gpu.h>

#include <stdio.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

namespace nova {

__global__ void pomdp_pbvi_initialize_alphaBA_gpu(unsigned int n, unsigned int m,
    unsigned int r, const float *R, float *alphaBA)
{
    unsigned int beliefIndex = blockIdx.x;
    unsigned int action = blockIdx.y;

    if (beliefIndex >= r || action >= m) {
        return;
    }

    // Compute Gamma_{a,*} and set it to the first value of alphaBA. Stride here.
    for (unsigned int s = threadIdx.x; s < n; s += blockDim.x) {
        alphaBA[beliefIndex * m * n + action * n + s] = R[s * m + action];
    }
}


__global__ void pomdp_pbvi_compute_alphaBA_gpu(unsigned int n, unsigned int ns, unsigned int m,
        unsigned int z, unsigned int r, unsigned int rz, float gamma,
        const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
        const float *Gamma, float *alphaBA)
{
    // Since float and unsigned int are 4 bytes each, and we need each array to be the size of
    // the number of threads, we will need to call this with:
    // sizeof(float) * numThreads + sizeof(unsigned int) * numThreads.
    // Note: blockDim.x == numThreads
    extern __shared__ float sdata[];
    float *maxAlphaDotBeta = (float *)sdata;
    unsigned int *maxAlphaIndex = (unsigned int *)&maxAlphaDotBeta[blockDim.x];

    maxAlphaDotBeta[threadIdx.x] = NOVA_FLT_MIN;
    maxAlphaIndex[threadIdx.x] = 0;

    __syncthreads();

    unsigned int beliefIndex = blockIdx.x;
    unsigned int action = blockIdx.y;
    unsigned int observation = blockIdx.z;

    if (beliefIndex >= r || action >= m || observation >= z) {
        return;
    }

    // Compute the max alpha vector from Gamma, given the fixed action and observation.
    // Note: this is the max w.r.t. just the strided elements. The reduction will
    // be computed afterwards for the max over all alpha-vectors.
    for (unsigned int alphaIndex = threadIdx.x; alphaIndex < r; alphaIndex += blockDim.x) {
        float alphaDotBeta = 0.0f;

        for (unsigned int i = 0; i < rz; i++) {
            int s = Z[beliefIndex * rz + i];
            if (s < 0) {
                break;
            }

            // We compute the value of this state in the alpha-vector, then multiply it by the
            // belief, and add it to the current dot product value for this alpha-vector.
            float value = 0.0f;
            for (unsigned int j = 0; j < ns; j++) {
                int sp = S[s * m * ns + action * ns + j];
                if (sp < 0) {
                    break;
                }
                value += T[s * m * ns + action * ns + j] *
                            O[action * n * z + sp * z + observation] *
                            Gamma[alphaIndex * n + sp];
            }

            __syncthreads();

            value *= gamma;

            alphaDotBeta += value * B[beliefIndex * rz + i];
        }

        __syncthreads();

        // Store the maximal value and index.
        if (alphaIndex == threadIdx.x || alphaDotBeta > maxAlphaDotBeta[threadIdx.x]) {
            maxAlphaDotBeta[threadIdx.x] = alphaDotBeta;
            maxAlphaIndex[threadIdx.x] = alphaIndex;
        }
    }

    // Note: The above code essentially does the first add during load. It takes care of *all*
    // the other elements *outside* the number of threads we have. In other words, starting here,
    // we already have computed part of the maxAlphaDotBeta and maxAlphaIndex; we just need to
    // finish the rest quickly, using a reduction.
    __syncthreads();

    // Use reduction to compute the max overall alpha-vector.
    for (unsigned int alphaIndex = blockDim.x / 2; alphaIndex > 0; alphaIndex >>= 1) {
        if (threadIdx.x < alphaIndex && threadIdx.x < r && threadIdx.x + alphaIndex < r) {
            if (maxAlphaDotBeta[threadIdx.x] < maxAlphaDotBeta[threadIdx.x + alphaIndex]) {
                maxAlphaDotBeta[threadIdx.x] = maxAlphaDotBeta[threadIdx.x + alphaIndex];
                maxAlphaIndex[threadIdx.x] = maxAlphaIndex[threadIdx.x + alphaIndex];
            }
        }

        __syncthreads();
    }

    // Now we can compute the alpha-vector component for this observation, since we have the max.
    // We will need to compute the dot product anyway, so let's just distribute the belief over the
    // sum over observations, and add it all up here.
    // Note: This re-uses the thread to stride over states now.
    for (unsigned int s = threadIdx.x; s < n; s += blockDim.x) {
        // We compute the value of this state in the alpha-vector, then multiply it by the belief,
        // and add it to the current dot product value for this alpha-vector.
        float value = 0.0f;
        for (unsigned int i = 0; i < ns; i++) {
            int sp = S[s * m * ns + action * ns + i];
            if (sp < 0) {
                break;
            }
            // Note: maxAlphaIndex[0] holds the maximal index value computed from the reduction.
            value += T[s * m * ns + action * ns + i] *
                        O[action * n * z + sp * z + observation] *
                        Gamma[maxAlphaIndex[0] * n + sp];
        }

        __syncthreads();

        alphaBA[beliefIndex * m * n + action * n + s] += gamma * value;
    }
}


__global__ void pomdp_pbvi_update_step_gpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
        unsigned int r, unsigned int rz, float gamma,
        const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
        const float *Gamma,
        float *alphaBA, float *GammaPrime, unsigned int *piPrime)
{
    // Each block will run a different belief. Our overall goal: Compute the value
    // of GammaPrime[beliefIndex * n + ???] and piPrime[beliefIndex].
    unsigned int beliefIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (beliefIndex >= r) {
        return;
    }

    // We want to find the action that maximizes the value, store it in piPrime, as well as
    // its alpha-vector GammaPrime.
    float maxActionValue = NOVA_FLT_MIN;

    for (unsigned int action = 0; action < m; action++) {
        // Only execute if the action is available.
        //if (available[beliefIndex * m + action]) {

        // The potential alpha-vector has been computed, so compute the value with respect
        // to the belief state.
        float actionValue = 0.0f;
        for (unsigned int i = 0; i < rz; i++) {
            int s = Z[beliefIndex * rz + i];
            if (s < 0) {
                break;
            }
            actionValue += alphaBA[beliefIndex * m * n + action * n + s] * B[beliefIndex * rz + i];
        }

        // If this was larger, then overwrite piPrime and GammaPrime's values.
        if (actionValue > maxActionValue) {
            maxActionValue = actionValue;

            piPrime[beliefIndex] = action;
        }

        //}
        //__syncthreads();
    }

    memcpy(&GammaPrime[beliefIndex * n], &alphaBA[beliefIndex * m * n + piPrime[beliefIndex] * n], n * sizeof(float));
    //for (unsigned int s = 0; s < n; s++) {
    //    GammaPrime[beliefIndex * n + s] = alphaBA[beliefIndex * m * n + piPrime[beliefIndex] * n + s];
    //}
}


int pomdp_pbvi_initialize_gpu(const POMDP *pomdp, POMDPPBVIGPU *pbvi)
{
    if (pomdp == nullptr || pomdp->r == 0 || pomdp->n == 0 || pbvi == nullptr) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon.
    pbvi->currentHorizon = 0;

    // Allocate the memory for the device-side Gamma, GammaPrime, pi, and the intermediate variable alphaBA.
    if (cudaMalloc(&pbvi->d_Gamma, pomdp->r * pomdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for Gamma.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMalloc(&pbvi->d_GammaPrime, pomdp->r * pomdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for Gamma (prime).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMalloc(&pbvi->d_pi, pomdp->r * sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for pi.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMalloc(&pbvi->d_alphaBA, pomdp->r * pomdp->m * pomdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for alphaBA.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // If GammaInitial is not provided, then create it. By default it takes 0.
    bool createdDefaultGammaInitial = false;

    if (pbvi->GammaInitial == nullptr) {
        pbvi->GammaInitial = new float[pomdp->r * pomdp->n];
        for (unsigned int i = 0; i < pomdp->r * pomdp->n; i++) {
            pbvi->GammaInitial[i] = 0.0f;
        }
        createdDefaultGammaInitial = true;
    }

    // Copy the GammaInitial data to Gamma and GammaPrime.
    if (cudaMemcpy(pbvi->d_Gamma, pbvi->GammaInitial, pomdp->r * pomdp->n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to copy memory from host to device for Gamma.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }
    if (cudaMemcpy(pbvi->d_GammaPrime, pbvi->GammaInitial, pomdp->r * pomdp->n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to copy memory from host to device for Gamma (prime).");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    // If we had created a default GammaInitial, then free it here.
    if (createdDefaultGammaInitial) {
        delete [] pbvi->GammaInitial;
        pbvi->GammaInitial = nullptr;
    }

    // Lastly, create a temporary variable to assign the default values of pi.
    unsigned int *pi = new unsigned int[pomdp->r];
    for (unsigned int i = 0; i < pomdp->r; i++) {
        pi[i] = 0;
    }

    if (cudaMemcpy(pbvi->d_pi, pi, pomdp->r * sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to copy memory from host to device for pi.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    delete [] pi;
    pi = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_pbvi_execute_gpu(const POMDP *pomdp, POMDPPBVIGPU *pbvi, POMDPAlphaVectors *policy)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->d_S == nullptr || pomdp->d_T == nullptr || pomdp->d_O == nullptr || pomdp->d_R == nullptr ||
            pomdp->d_Z == nullptr || pomdp->d_B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            pbvi == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_pbvi_execute_gpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Ensure threads are correct.
    if (pbvi->numThreads % 32 != 0) {
        fprintf(stderr, "Error[pomdp_pbvi_execute_gpu]: %s\n", "Invalid number of threads.");
        return NOVA_ERROR_INVALID_CUDA_PARAM;
    }

    int result = pomdp_pbvi_initialize_gpu(pomdp, pbvi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_pbvi_execute_gpu]: %s\n", "Failed to initialize GPU variables.");
        return result;
    }

    // For each of the updates, run PBVI. Note that the currentHorizon is initialized to zero
    // above, and is updated in the update function below.
    while (pbvi->currentHorizon < pomdp->horizon) {
        //printf("PBVI (GPU Version) -- Iteration %i of %i\n", pomdp->currentHorizon, pomdp->horizon);

        result = pomdp_pbvi_update_gpu(pomdp, pbvi);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[pomdp_pbvi_execute_gpu]: %s\n", "Failed to perform PBVI update step.");
            return result;
        }
    }

    result = pomdp_pbvi_get_policy_gpu(pomdp, pbvi, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_pbvi_execute_gpu]: %s\n", "Failed to get the policy.");
    }

    result = pomdp_pbvi_uninitialize_gpu(pomdp, pbvi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_pbvi_execute_gpu]: %s\n", "Failed to uninitialize the GPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int pomdp_pbvi_uninitialize_gpu(const POMDP *pomdp, POMDPPBVIGPU *pbvi)
{
    if (pomdp == nullptr || pbvi == nullptr) {
        fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = NOVA_SUCCESS;

    // Reset the current horizon.
    pbvi->currentHorizon = 0;

    if (pbvi->d_Gamma != nullptr) {
        if (cudaFree(pbvi->d_Gamma) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s\n",
                    "Failed to allocate device-side memory for the Gamma (the alpha-vectors).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    pbvi->d_Gamma = nullptr;

    if (pbvi->d_GammaPrime != nullptr) {
        if (cudaFree(pbvi->d_GammaPrime) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s\n",
                    "Failed to allocate device-side memory for the GammaPrime (the alpha-vectors' copy).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    pbvi->d_GammaPrime = nullptr;

    if (pbvi->d_pi != nullptr) {
        if (cudaFree(pbvi->d_pi) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s\n",
                    "Failed to allocate device-side memory for the pi (the policy).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    pbvi->d_pi = nullptr;

    if (pbvi->d_alphaBA != nullptr) {
        if (cudaFree(pbvi->d_alphaBA) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s\n",
                    "Failed to allocate device-side memory for alphaBA (alpha-vector collection).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    pbvi->d_alphaBA = nullptr;

    return result;
}


int pomdp_pbvi_update_gpu(const POMDP *pomdp, POMDPPBVIGPU *pbvi)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->d_S == nullptr || pomdp->d_T == nullptr || pomdp->d_O == nullptr || pomdp->d_R == nullptr ||
            pomdp->d_Z == nullptr || pomdp->d_B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            pbvi == nullptr || pbvi->d_Gamma == nullptr || pbvi->d_GammaPrime == nullptr ||
            pbvi->d_pi == nullptr || pbvi->d_alphaBA == nullptr) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Initialize the alphaBA with default values of the rewards R(*, a).
    pomdp_pbvi_initialize_alphaBA_gpu<<< dim3(pomdp->r, pomdp->m, 1), pbvi->numThreads >>>(
                                            pomdp->n, pomdp->m, pomdp->r, pomdp->d_R,
                                            pbvi->d_alphaBA);

    // Check if there was an error executing the kernel.
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                        "Failed to execute the 'initialization of alphaBA' kernel.");
        return NOVA_ERROR_KERNEL_EXECUTION;
    }

    // Wait for the kernel to finish before looping more.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                    "Failed to synchronize the device after 'initialization of alphaBA' kernel.");
        return NOVA_ERROR_DEVICE_SYNCHRONIZE;
    }

    pomdp_pbvi_compute_alphaBA_gpu<<< dim3(pomdp->r, pomdp->m, pomdp->z),
                                    pbvi->numThreads,
                                    pbvi->numThreads * sizeof(float) +
                                    pbvi->numThreads * sizeof(unsigned int) >>>(
                                pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                                pomdp->d_S, pomdp->d_T, pomdp->d_O, pomdp->d_R, pomdp->d_Z, pomdp->d_B,
                                pbvi->d_Gamma,
                                pbvi->d_alphaBA);

    // Check if there was an error executing the kernel.
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                        "Failed to execute the 'compute_alphaBA' kernel.");
        return NOVA_ERROR_KERNEL_EXECUTION;
    }

    // Wait for the kernel to finish before looping more.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                        "Failed to synchronize the device after 'compute_alphaBA' kernel.");
        return NOVA_ERROR_DEVICE_SYNCHRONIZE;
    }

    // Compute the number of blocks.
    int numBlocks = (unsigned int)((float)pomdp->r / (float)pbvi->numThreads) + 1;

    // Execute a kernel for the first three stages of for-loops: B, A, Z, as a 3d-block,
    // and the 4th stage for-loop over Gamma as the threads.
    if (pbvi->currentHorizon % 2 == 0) {
        pomdp_pbvi_update_step_gpu<<< numBlocks, pbvi->numThreads >>>(
                pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                pomdp->d_S, pomdp->d_T, pomdp->d_O, pomdp->d_R, pomdp->d_Z, pomdp->d_B, pbvi->d_Gamma,
                pbvi->d_alphaBA, pbvi->d_GammaPrime, pbvi->d_pi);
    } else {
        pomdp_pbvi_update_step_gpu<<< numBlocks, pbvi->numThreads >>>(
                pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                pomdp->d_S, pomdp->d_T, pomdp->d_O, pomdp->d_R, pomdp->d_Z, pomdp->d_B, pbvi->d_GammaPrime,
                pbvi->d_alphaBA, pbvi->d_Gamma, pbvi->d_pi);
    }

    // Check if there was an error executing the kernel.
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                        "Failed to execute the 'pomdp_pbvi_update_step_gpu' kernel.");
        return NOVA_ERROR_KERNEL_EXECUTION;
    }

    // Wait for the kernel to finish before looping more.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                        "Failed to synchronize the device after 'pomdp_pbvi_update_step_gpu' kernel.");
        return NOVA_ERROR_DEVICE_SYNCHRONIZE;
    }

    pbvi->currentHorizon++;

    return NOVA_SUCCESS;
}


int pomdp_pbvi_get_policy_gpu(const POMDP *pomdp, POMDPPBVIGPU *pbvi, POMDPAlphaVectors *policy)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->m == 0 || pomdp->r == 0 ||
            pbvi == nullptr || (pbvi->currentHorizon % 2 == 0 && pbvi->d_Gamma == nullptr) ||
            (pbvi->currentHorizon % 2 == 1 && pbvi->d_GammaPrime == nullptr) || pbvi->d_pi == nullptr ||
            policy == nullptr) {
        fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Initialize the policy, which allocates memory.
    int result = pomdp_alpha_vectors_initialize(policy, pomdp->n, pomdp->m, pomdp->r);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Copy the final result of Gamma and pi to the variables provided, from device to host.
    // This assumes that the memory has been allocated for the variables provided.
    if (pbvi->currentHorizon % 2 == 0) {
        if (cudaMemcpy(policy->Gamma, pbvi->d_Gamma, pomdp->r * pomdp->n * sizeof(float),
                        cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s\n",
                    "Failed to copy memory from device to host for Gamma.");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    } else {
        if (cudaMemcpy(policy->Gamma, pbvi->d_GammaPrime, pomdp->r * pomdp->n * sizeof(float),
                        cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s\n",
                    "Failed to copy memory from device to host for Gamma (prime).");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    }

    if (cudaMemcpy(policy->pi, pbvi->d_pi, pomdp->r * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s\n",
                "Failed to copy memory from device to host for pi.");
        return NOVA_ERROR_MEMCPY_TO_HOST;
    }

    return NOVA_SUCCESS;
}

}; // namespace nova

