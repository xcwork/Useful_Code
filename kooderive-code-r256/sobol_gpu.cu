// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3
// note derived from code provided by NVIDIA 

// NVIDIA licensing terms: 
//Source Code: Developer shall have the right to modify and create derivative works with the Source Code.
//Developer shall own any derivative works ("Derivatives") it creates to the Source Code, provided that
//Developer uses the Materials in accordance with the terms and conditions of this Agreement. Developer
//may distribute the Derivatives, provided that all NVIDIA copyright notices and trademarks are used properly
//and the Derivatives include the following statement: "This software contains source code provided by
//NVIDIA Corporation."



/*
 * Sobol Quasi-random Number Generator example
 *
 * Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
 * http://people.maths.ox.ac.uk/~gilesm/
 *
 * and C code developed by Stephen Joe, University of Waikato, New Zealand
 * and Frances Kuo, University of New South Wales, Australia
 * http://web.maths.unsw.edu.au/~fkuo/sobol/
 *
 * For theoretical background see:
 *
 * P. Bratley and B.L. Fox.
 * Implementing Sobol's quasirandom sequence generator
 * http://portal.acm.org/citation.cfm?id=42288
 * ACM Trans. on Math. Software, 14(1):88-100, 1988
 *
 * S. Joe and F. Kuo.
 * Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
 * http://portal.acm.org/citation.cfm?id=641879
 * ACM Trans. on Math. Software, 29(1):49-57, 2003
 *
 * NOTICE TO USER:
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include "sobol.h"
#include "sobol_gpu.h"

#define k_2powneg32 2.3283064E-10F

__global__ void sobolGPU_kernel(unsigned n_vectors, unsigned n_dimensions, unsigned path_offset, unsigned *d_directions, float *d_output)
{
    __shared__ unsigned int v[n_directions];

    // Offset into the correct dimension as specified by the
    // block y coordinate
    d_directions = d_directions + n_directions * blockIdx.y;
    d_output = d_output +  n_vectors * blockIdx.y;

    // Copy the direction numbers for this dimension into shared
    // memory - there are only 32 direction numbers so only the
    // first 32 (n_directions) threads need participate.
    if (threadIdx.x < n_directions)
    {
	    v[threadIdx.x] = d_directions[threadIdx.x];
    }
    __syncthreads();

    // Set initial index (i.e. which vector this thread is
    // computing first) and stride (i.e. step to the next vector
    // for this thread)
        
    int outputi0= threadIdx.x + blockIdx.x * blockDim.x;

    int i0     = outputi0+path_offset;
    int stride = gridDim.x * blockDim.x;

    // Get the gray code of the index
    // c.f. Numerical Recipes in C, chapter 20
    // http://www.nrbook.com/a/bookcpdf/c20-2.pdf
    unsigned int g = i0 ^ (i0 >> 1);

    // Initialisation for first point x[i0]
    // In the Bratley and Fox paper this is equation (*), where
    // we are computing the value for x[n] without knowing the
    // value of x[n-1].
    unsigned int X = 0;
    unsigned int mask;
    for (unsigned int k = 0 ; k < __ffs(stride) - 1 ; k++)
    {
        // We want X ^= g_k * v[k], where g_k is one or zero.
        // We do this by setting a mask with all bits equal to
        // g_k. In reality we keep shifting g so that g_k is the
        // LSB of g. This way we avoid multiplication.
        mask = - (g & 1);
        X ^= mask & v[k];
        g = g >> 1;
    }
    if (outputi0 < n_vectors)
    {
        d_output[i0-path_offset] = (float)X * k_2powneg32;
    }

    // Now do rest of points, using the stride
    // Here we want to generate x[i] from x[i-stride] where we
    // don't have any of the x in between, therefore we have to
    // revisit the equation (**), this is easiest with an example
    // so assume stride is 16.
    // From x[n] to x[n+16] there will be:
    //   8 changes in the first bit
    //   4 changes in the second bit
    //   2 changes in the third bit
    //   1 change in the fourth
    //   1 change in one of the remaining bits
    //
    // What this means is that in the equation:
    //   x[n+1] = x[n] ^ v[p]
    //   x[n+2] = x[n+1] ^ v[q] = x[n] ^ v[p] ^ v[q]
    //   ...
    // We will apply xor with v[1] eight times, v[2] four times,
    // v[3] twice, v[4] once and one other direction number once.
    // Since two xors cancel out, we can skip even applications
    // and just apply xor with v[4] (i.e. log2(16)) and with
    // the current applicable direction number.
    // Note that all these indices count from 1, so we need to
    // subtract 1 from them all to account for C arrays counting
    // from zero.
    unsigned int v_log2stridem1 = v[__ffs(stride) - 2];
    unsigned int v_stridemask = stride - 1;
    for (unsigned int i = i0 + stride ; i < n_vectors ; i += stride)
    {
        // x[i] = x[i-stride] ^ v[b] ^ v[c]
        //  where b is log2(stride) minus 1 for C array indexing
        //  where c is the index of the rightmost zero bit in i,
        //  not including the bottom log2(stride) bits, minus 1
        //  for C array indexing
        // In the Bratley and Fox paper this is equation (**)
        X ^= v_log2stridem1 ^ v[__ffs(~((i - stride) | v_stridemask)) - 1];
        d_output[i-path_offset] = (float)X * k_2powneg32;
    }
}



__device__
float invCumGiles(const float& x) 
{
  const float POLY[] = { 4.560330337e-12f ,   1.891879037e-09f ,
                 6.384940120e-11f ,   2.112199920e-08f ,
                 3.709377697e-10f ,  -4.363333596e-09f ,
                 1.235508984e-09f ,  -1.340785237e-06f ,
                -2.085483499e-08f ,  -7.816014247e-06f ,
                -8.052312355e-07f ,   2.418058136e-05f ,
                -1.789424115e-05f ,   0.0005984976896f ,
                -0.0003717238455f ,   0.003071246478f ,
                -0.009026965781f  ,  -0.01373818033f ,
                -0.4222077998f    ,  -0.6926406482f ,
                 9.066901046f     ,   4.381417061f     };

 float MEAN[] = { -11.36390346f ,  -2.693891179f };
 float MID    = -6.114811502f;
 float w, p;
 int   n;

 w = __logf(4.0f*x*(1.0f-x));
 n = w>MID;
 w = w - MEAN[n];

 p = POLY[n];
 for (; n<22; n+=2)
  p = p*w + POLY[n];

 return (x-0.5f)*p;
}


#define CQP(v) (P1+v*(P2+v*(P3+v*(P4+(P5+P6*v)*v))))
#define CQQ(v) (Q1+v*(Q2+v*(Q3+v*(Q4+(Q5+Q6*v)*v))))
__device__
float ICNDfloat2(float v)
{
const float P1 = 1.2533136835212087879;
const float P2 = 1.9797154223229267471;
const float P3 = 0.80002295072483916762;
const float P4 = 0.087403248265958578062;
const float P5 = 0.0020751409553756572917;
const float P6 = 4.744820732427972462e-6;
const float Q1 = 1.0;
const float Q2 = 2.0795584360534589311;
const float Q3 = 1.2499328117341603014;
const float Q4 = 0.23668431621373705623;
const float Q5 = 0.0120098270559197768;
const float Q6 = 0.00010590620919921025259;
float z;
int sgn;
sgn = (v >= 0.5f);
sgn = sgn - !sgn;
z = -logf(1.0f - (sgn * ((2.0f * v) - 1.0f)));
return sgn * z * CQP(z) / CQQ(z);
}



__global__ void sobolGPU_kernel_normals(unsigned n_vectors, unsigned n_dimensions, unsigned path_offset, unsigned *d_directions, float *d_output)
{
    __shared__ unsigned int v[n_directions];

    // Offset into the correct dimension as specified by the
    // block y coordinate
    d_directions = d_directions + n_directions * blockIdx.y;
    d_output = d_output +  n_vectors * blockIdx.y;

    // Copy the direction numbers for this dimension into shared
    // memory - there are only 32 direction numbers so only the
    // first 32 (n_directions) threads need participate.
    if (threadIdx.x < n_directions)
    {
	    v[threadIdx.x] = d_directions[threadIdx.x];
    }
    __syncthreads();

    // Set initial index (i.e. which vector this thread is
    // computing first) and stride (i.e. step to the next vector
    // for this thread)
    int i0     = threadIdx.x + blockIdx.x * blockDim.x+path_offset;
    int i0out = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    // Get the gray code of the index
    // c.f. Numerical Recipes in C, chapter 20
    // http://www.nrbook.com/a/bookcpdf/c20-2.pdf
    unsigned int g = i0 ^ (i0 >> 1);

    // Initialisation for first point x[i0]
    // In the Bratley and Fox paper this is equation (*), where
    // we are computing the value for x[n] without knowing the
    // value of x[n-1].
    unsigned int X = 0;
    unsigned int mask;
    for (unsigned int k = 0 ; k < __ffs(stride) - 1 ; k++)
    {
        // We want X ^= g_k * v[k], where g_k is one or zero.
        // We do this by setting a mask with all bits equal to
        // g_k. In reality we keep shifting g so that g_k is the
        // LSB of g. This way we avoid multiplication.
        mask = - (g & 1);
        X ^= mask & v[k];
        g = g >> 1;
    }
    if (i0 < n_vectors)
    {
        d_output[i0out] = ICNDfloat2((float)X * k_2powneg32);
    }

    // Now do rest of points, using the stride
    // Here we want to generate x[i] from x[i-stride] where we
    // don't have any of the x in between, therefore we have to
    // revisit the equation (**), this is easiest with an example
    // so assume stride is 16.
    // From x[n] to x[n+16] there will be:
    //   8 changes in the first bit
    //   4 changes in the second bit
    //   2 changes in the third bit
    //   1 change in the fourth
    //   1 change in one of the remaining bits
    //
    // What this means is that in the equation:
    //   x[n+1] = x[n] ^ v[p]
    //   x[n+2] = x[n+1] ^ v[q] = x[n] ^ v[p] ^ v[q]
    //   ...
    // We will apply xor with v[1] eight times, v[2] four times,
    // v[3] twice, v[4] once and one other direction number once.
    // Since two xors cancel out, we can skip even applications
    // and just apply xor with v[4] (i.e. log2(16)) and with
    // the current applicable direction number.
    // Note that all these indices count from 1, so we need to
    // subtract 1 from them all to account for C arrays counting
    // from zero.
    unsigned int v_log2stridem1 = v[__ffs(stride) - 2];
    unsigned int v_stridemask = stride - 1;
    for (unsigned int i = i0 + stride ; i < n_vectors ; i += stride)
    {
        // x[i] = x[i-stride] ^ v[b] ^ v[c]
        //  where b is log2(stride) minus 1 for C array indexing
        //  where c is the index of the rightmost zero bit in i,
        //  not including the bottom log2(stride) bits, minus 1
        //  for C array indexing
        // In the Bratley and Fox paper this is equation (**)
        X ^= v_log2stridem1 ^ v[__ffs(~((i - stride) | v_stridemask)) - 1];
        
        d_output[i-path_offset] = ICNDfloat2((float)X * k_2powneg32);
//        invCumGiles( (float)X * k_2powneg32);
    }
}


extern "C"
void sobolGPU(int n_vectors, int n_dimensions, unsigned path_offset, unsigned int *d_directions, float *d_output, bool inverseCum)
{
    const int threadsperblock = 64;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    // This implementation of the generator outputs all the draws for
    // one dimension in a contiguous region of memory, followed by the
    // next dimension and so on.
    // Therefore all threads within a block will be processing different
    // vectors from the same dimension. As a result we want the total
    // number of blocks to be a multiple of the number of dimensions.
    dimGrid.y = n_dimensions;

    // If the number of dimensions is large then we will set the number
    // of blocks to equal the number of dimensions (i.e. dimGrid.x = 1)
    // but if the number of dimensions is small (e.g. less than 32) then
    // we'll partition the vectors across blocks (as well as threads).
    // We also need to cap the dimGrid.x where the number of vectors
    // is too small to be partitioned.
    dimGrid.x = 1 + 31 / n_dimensions;
    if (dimGrid.x > (unsigned int)(n_vectors / threadsperblock))
    {
        dimGrid.x = (n_vectors + threadsperblock - 1) / threadsperblock;
    }
    
    // Fix the number of threads
    dimBlock.x = threadsperblock;

    // Execute GPU kernel
    if (inverseCum)
        sobolGPU_kernel_normals<<<dimGrid, dimBlock>>>(n_vectors, n_dimensions, path_offset,d_directions, d_output);
    else
         sobolGPU_kernel<<<dimGrid, dimBlock>>>(n_vectors, n_dimensions, path_offset,d_directions, d_output);
}

