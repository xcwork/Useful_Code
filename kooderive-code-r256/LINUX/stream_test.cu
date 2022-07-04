//
//
//                          Stream_test.cu
//
//
// (c) Mark Joshi 2011,2014
// This code is released under the GNU public licence version 3

#include "stream_test.h"
#include <cudaWrappers/vector_pinned.h>
#include <cudaWrappers/cudaEventWrapper.h>
#include <cudaWrappers/cudaStreamWrapper.h>
#include <thrust/device_vector.h>
#include <cutil.h>
#include <cutil_inline.h>

namespace
{

    __host__ __device__ int testFunction(int a, int b)
    {
       int c= (a+1)/(b+1);

        return c;
    }

}    __global__ void streamtestkernel(int *a , int *b, int* c,int N)
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;

        if (idx < N)
        {


            c[idx] = testFunction(a[idx],b[idx]);
        }
    }

namespace
{
    int SingleStreamTest(bool verbose, size_t N, size_t M,DeviceChooser& chooser)
    {
        std::cout << " Stream test " << N << " " << M << "\n";
        int errors=0;

        {

            size_t allDataSize = N*M;
            vector_pinned<int> host_a(allDataSize);
            vector_pinned<int> host_b(allDataSize);
            vector_pinned<int> host_c(allDataSize);


            cudaSetDevice(chooser.WhichDevice());


            cudaThreadSynchronize();



            thrust::device_vector<int> dev_a(N);
            thrust::device_vector<int> dev_b(N);
            thrust::device_vector<int> dev_c(N);

            cudaStreamWrapper streamWrapper;

            for (size_t i=0; i < allDataSize; ++i)
            {
                host_a[i] = i;
                host_b[i] = allDataSize-i;
            }

            cudaEventWrapper event1;
            cudaEventWrapper event2;
            event1.record();

            int i=0;

            for ( i=0; i < static_cast<int>(allDataSize); i+=N)
            {
                cutilSafeCall( cudaMemcpyAsync(thrust::raw_pointer_cast(&dev_a[0]),&host_a[i],N*sizeof(int),cudaMemcpyHostToDevice,
                    *streamWrapper));

                cutilSafeCall( cudaMemcpyAsync(thrust::raw_pointer_cast(&dev_b[0]),&host_b[i],N*sizeof(int),cudaMemcpyHostToDevice,
                    *streamWrapper));  

                streamtestkernel<<<N/256+1,256,0,*streamWrapper>>>(thrust::raw_pointer_cast(&dev_a[0]),
                    thrust::raw_pointer_cast(&dev_b[0]),
                    thrust::raw_pointer_cast(&dev_c[0]),
                    static_cast<int>(N));

                cutilSafeCall( cudaMemcpyAsync(&host_c[i],thrust::raw_pointer_cast(&dev_c[0]),N*sizeof(int),cudaMemcpyDeviceToHost,
                    *streamWrapper));  


            }
            cutilSafeCall(streamWrapper.synchronize());
            event2.record();
            event2.synchronize();



            float timeTaken =   event2.timeSince(event1).first;
            std::cout << "\n time taken " << timeTaken << "\n";




            for (size_t j=0; j < allDataSize; ++j)
                if (testFunction(host_a[j] , host_b[j] )!= host_c[j])
                {
                    ++errors;
                    std::cout << host_a[j] << " " << host_b[j]  << " " << testFunction(host_a[j] , host_b[j] ) << " " << host_c[j] << "\n";
                }


        }
        cudaThreadExit();

        if (verbose)
            if (errors >0)
                std::cout << "Test Failed. Errors found : " << errors << "\n";
            else
                std::cout << "one stream test passed\n";

        return errors > 0 ? 1 : 0;
    }


    int TwoStreamTest(bool verbose, size_t N, size_t M,DeviceChooser& chooser)
    {
        std::cout << "Two Stream test " << N << " " << M << "\n";
        int errors=0;

        {

            size_t allDataSize = N*M;
            vector_pinned<int> host_a(allDataSize);
            vector_pinned<int> host_b(allDataSize);
            vector_pinned<int> host_c(allDataSize);


            cudaSetDevice(chooser.WhichDevice());


            cudaThreadSynchronize();



            thrust::device_vector<int> dev_a(N);
            thrust::device_vector<int> dev_b(N);
            thrust::device_vector<int> dev_c(N);

            thrust::device_vector<int> dev_a2(N);
            thrust::device_vector<int> dev_b2(N);
            thrust::device_vector<int> dev_c2(N);


            cudaStreamWrapper streamWrapper;
            cudaStreamWrapper streamWrapper2;    

            for (size_t i=0; i < allDataSize; ++i)
            {
                host_a[i] = i;
                host_b[i] = allDataSize-i;
            }

            cudaEventWrapper event1;
            cudaEventWrapper event2;
            event1.record();

            int i=0;

            for ( i=0; i < static_cast<int>(allDataSize); i+=2*N)
            {
                cutilSafeCall( cudaMemcpyAsync(thrust::raw_pointer_cast(&dev_a[0]),&host_a[i],N*sizeof(int),cudaMemcpyHostToDevice,
                    *streamWrapper));

                cutilSafeCall( cudaMemcpyAsync(thrust::raw_pointer_cast(&dev_b[0]),&host_b[i],N*sizeof(int),cudaMemcpyHostToDevice,
                    *streamWrapper));  

                cutilSafeCall( cudaMemcpyAsync(thrust::raw_pointer_cast(&dev_a2[0]),&host_a[N+i],N*sizeof(int),cudaMemcpyHostToDevice,
                    *streamWrapper2));

                cutilSafeCall( cudaMemcpyAsync(thrust::raw_pointer_cast(&dev_b2[0]),&host_b[N+i],N*sizeof(int),cudaMemcpyHostToDevice,
                    *streamWrapper2));  

		//		std::cout << " inserting test kernel into stream 1\n";

                streamtestkernel<<<N/256+1,256,0,*streamWrapper>>>(thrust::raw_pointer_cast(&dev_a[0]),
                    thrust::raw_pointer_cast(&dev_b[0]),
                    thrust::raw_pointer_cast(&dev_c[0]),
                    static_cast<int>(N));

				
	//			std::cout << " inserting test kernel into stream 2\n";

                streamtestkernel<<<N/256+1,256,0,*streamWrapper2>>>(thrust::raw_pointer_cast(&dev_a2[0]),
                    thrust::raw_pointer_cast(&dev_b2[0]),
                    thrust::raw_pointer_cast(&dev_c2[0]),
                    static_cast<int>(N));


                cutilSafeCall( cudaMemcpyAsync(&host_c[i],thrust::raw_pointer_cast(&dev_c[0]),N*sizeof(int),cudaMemcpyDeviceToHost,
                    *streamWrapper));  

                cutilSafeCall( cudaMemcpyAsync(&host_c[i+N],thrust::raw_pointer_cast(&dev_c2[0]),N*sizeof(int),cudaMemcpyDeviceToHost,
                    *streamWrapper2));

            }
            cutilSafeCall(streamWrapper.synchronize());
            cutilSafeCall(streamWrapper2.synchronize());
			event2.record();
            event2.synchronize();



            float timeTaken =   event2.timeSince(event1).first;
            std::cout << "\n time taken " << timeTaken << "\n";




            for (size_t j=0; j < allDataSize; ++j)
                if (testFunction(host_a[j] , host_b[j] )!= host_c[j])
                    ++errors;

        }
        cudaThreadExit();

        if (verbose)
            if  (errors >0)
                std::cout << "Stream test failed. errors found : " << errors << "\n";
            else
                std::cout << "Stream test passed\n";


        return errors > 0 ? 1 : 0;
    }
}


int StreamTestRoutine(bool Verbose,DeviceChooser& chooser)
{
    size_t N = 100000;
    size_t M =10;


    	   size_t estimatedFloatsRequired = 4*N*M;
   
         size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;
  
       cudaSetDevice(chooser.WhichDevice());

       ConfigCheckForGPU checker;

       float excessMem=2.0f;

       while (excessMem >0.0f)
       {
           checker.checkGlobalMem( estimatedMem, excessMem );

           if (excessMem >0.0f)
           {
               N/=2;
             
    	       estimatedFloatsRequired = 4*N*M;
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
     
           }
       }

     cudaThreadExit();

    int f=0;

    f+= TwoStreamTest(Verbose,N,M,chooser);

    
    f+= SingleStreamTest(Verbose,N,M,chooser);

    N=10000;
    M=100;

    f+= SingleStreamTest(Verbose,N,M,chooser);
    f+= TwoStreamTest(Verbose,N,M,chooser);

    N=1000000;
    M=2;

    f+= SingleStreamTest(Verbose,N,M,chooser);
    f+= TwoStreamTest(Verbose,N,M,chooser);
 
    return f;
}
