//               multithreading.h

// (c) Mark Joshi 2013

// released under GPL v3

// derived from code carrying following licence

/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
Extract from EULA: 
Source Code:  Developer shall have the right to modify and create derivative works with the Source Code.  
Developer shall own any derivative works ("Derivatives") it creates to the Source Code, provided that
Developer uses the Materials in accordance with the terms and conditions of this Agreement.  Developer may
distribute the Derivatives, provided that all NVIDIA copyright notices and trademarks are propagated and 
used properly and the Derivatives include the following statement: 
"This software contains source code provided by NVIDIA Corporation."  


 */


#ifndef MULTITHREADING_H
#define MULTITHREADING_H


//Simple portable thread library.

#if _WIN32
//Windows threads.
#include <windows.h>

typedef HANDLE CUTThread;
typedef unsigned(WINAPI *CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC unsigned WINAPI
#define  CUT_THREADEND return 0

#else
//POSIX threads.
#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void
#define  CUT_THREADEND
#endif


#ifdef __cplusplus
extern "C" {
#endif

    //Create thread.
    CUTThread cutStartThread(CUT_THREADROUTINE, void *data);

    //Wait for thread to finish.
    void cutEndThread(CUTThread thread);

    //Destroy thread.
    void cutDestroyThread(CUTThread thread);

    //Wait for multiple threads.
    void cutWaitForThreads(const CUTThread *threads, int num);

#ifdef __cplusplus
} //extern "C"
#endif

#endif //MULTITHREADING_H
