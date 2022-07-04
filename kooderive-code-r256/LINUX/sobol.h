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

#ifndef SOBOL_H
#define SOBOL_H
#include <gold/pragmas.h> 
#include <gold/sobol_gold.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


//outputs host vector, calls SobolGPU
//int Sobmain(int n_vectors, int n_dimensions, unsigned path_offset,  thrust::host_vector<float>& h_outputGPU_vec, bool doInverseNormal );

// outputs host vector, calls SobDevice
int SobDeviceMain(int n_vectors,unsigned path_offset,  int n_dimensions, thrust::host_vector<float>& h_outputGPU_vec, bool doInverseNormal );

// outputs using Device vector floats
int SobDevice(int n_vectors, int n_dimensions,unsigned path_offset,  thrust::device_vector<float>& d_output_vec, bool doInverseNormal, bool verbose=true);

// outputs using Device vector unsigned ints
int SobDeviceInts(int n_vectors, int n_dimensions,unsigned path_offset,  thrust::device_vector<unsigned int>& d_output_device);

#endif
