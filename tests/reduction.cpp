/*
 * This file is part of hipCPU, a HIP implementation based on OpenMP
 *
 * Copyright (c) 2018,2019 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <hip/hip_runtime.h>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <string>
#include <iostream>

void check_error(hipError_t err)
{
  if(err != hipSuccess)
    throw std::runtime_error{"Caught hip error: "+std::to_string(static_cast<int>(err))};
}

template<class T>
__global__ void block_reduce(const T* in, T* block_sums_out, int num_elements)
{
#ifdef __HIPCPU__
  T* scratch = (T*)HIP_DYNAMIC_SHARED_MEMORY;
#else
  extern __shared__ T scratch[];
#endif

  int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int lid = hipThreadIdx_x;

  if(gid < num_elements)
    scratch[lid] = in[gid];

  for(int i = hipBlockDim_x/2; i > 0; i /= 2)
  {
    __syncthreads();

    if(lid < i)
      scratch[lid] += scratch[lid + i];
  }

  if(lid == 0)
    block_sums_out[hipBlockIdx_x] = scratch[0];
}
  
constexpr std::size_t buff_size = 1024;
constexpr std::size_t block_size = 16;

static_assert(buff_size % block_size == 0, 
              "buffer size must be a multiple of the block size");
  
constexpr std::size_t num_blocks = buff_size / block_size;

int main()
{
  std::vector<int> in(buff_size);
  std::vector<int> block_sums(num_blocks);
  
  for(std::size_t i = 0; i < buff_size; ++i)
    in[i] = i;
  
  int *d_in, *d_out;
  check_error(hipMalloc((void**)&d_in, buff_size * sizeof(int)));
  check_error(hipMalloc((void**)&d_out, num_blocks * sizeof(int)));
  
  check_error(hipMemcpy(d_in, in.data(), buff_size * sizeof(int), hipMemcpyHostToDevice));
  
  hipLaunchKernelGGL(block_reduce, num_blocks, block_size, block_size*sizeof(int), 0, 
                     d_in, d_out, buff_size);
  
  check_error(hipMemcpy(block_sums.data(), d_out, num_blocks * sizeof(int), hipMemcpyDeviceToHost));
  
  for(std::size_t i = 0; i < block_sums.size(); ++i)
  {
    int expected_sum = 0;
    for(int j = 0; j < block_size; ++j)
      expected_sum += in[i*block_size + j];
    assert(expected_sum == block_sums[i]);
  }
  
}
