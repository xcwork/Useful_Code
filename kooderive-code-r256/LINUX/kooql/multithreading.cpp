//               multithreading.cpp

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

#include "multithreading.h"

#if _WIN32
//Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data)
{
    return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, data, 0, NULL);
}

//Wait for thread to finish
void cutEndThread(CUTThread thread)
{
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}

//Destroy thread
void cutDestroyThread(CUTThread thread)
{
    TerminateThread(thread, 0);
    CloseHandle(thread);
}

//Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num)
{
    WaitForMultipleObjects(num, threads, true, INFINITE);

    for (int i = 0; i < num; i++)
    {
        CloseHandle(threads[i]);
    }
}

#else
//Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data)
{
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

//Wait for thread to finish
void cutEndThread(CUTThread thread)
{
    pthread_join(thread, NULL);
}

//Destroy thread
void cutDestroyThread(CUTThread thread)
{
    pthread_cancel(thread);
}

//Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num)
{
    for (int i = 0; i < num; i++)
    {
        cutEndThread(threads[i]);
    }
}

#endif
