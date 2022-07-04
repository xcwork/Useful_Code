/*
                                         CUDAConditionalMacros.h

                                         (c) Mark Joshi 2013,2014

Code released under GPI licence V 3.
*/

#ifndef CUDA_COND_MACROS_H
#define CUDA_COND_MACROS_H


#if __CUDA_ARCH__ >= 350
    #define LDG(x) __ldg(x)
#else
    #define LDG(x) *(x)
#endif

#endif
