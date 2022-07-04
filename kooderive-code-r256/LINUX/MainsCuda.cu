// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3


//   mainsCuda.cu


#include "Asian.h"

#include <cuda_runtime.h>

#include <cutil_inline.h>
#include "cudaMacros.h"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include <inverseCumulatives.h>
#include <gold/Bridge_gold.h>
#include "sobol.h"

#include "Bridge.h"


#include "Utilities.h"

int mainAsianRoutine(bool waitAtEnd)
{

    int path_offset=0;

    int loops = 1;
    std::vector<float> results(loops);

    int n_vectors =  intPower(2,15);
    int n_poweroftwo =8;
    int n_dimensions = intPower(2,n_poweroftwo);

    int N= n_vectors*n_dimensions; 
    bool doNormalinSobol = true;

    //cudaSetDevice(cutGetMaxGflopsDeviceId());


    cudaThreadSynchronize();

    int totalLoopTime = clock();


    for (int j=0; j < loops; ++j)
    {   

        int totalStart = clock();

        thrust::device_vector<float> dev_output(N);    
        thrust::device_vector<float> device_input(N);

        thrust::device_vector<float> outputDataArithmetic(n_vectors);   
        thrust::device_vector<float> outputDataGeometric(n_vectors);


        TimeSoFar(totalStart,"first allocs done");
        cutilSafeCall(cudaThreadSynchronize());

        cudaThreadSynchronize();

        CUT_CHECK_ERR("before Sobol \n");
        TimeSoFar(totalStart,"enter Sobol");

        if (     doNormalinSobol)
            int res =  SobDevice( n_vectors, n_dimensions,path_offset, device_input,doNormalinSobol);
        else
            int res =  SobDevice( n_vectors, n_dimensions, path_offset,dev_output,doNormalinSobol);


        cudaThreadSynchronize();

        CUT_CHECK_ERR("after Sobol \n");

        TimeSoFar(totalStart,"Sobol done");


        cudaThreadSynchronize();


        CUT_CHECK_ERR("before inverse\n");



        TimeSoFar(totalStart,"entering inverse cumulatives");

        if (!doNormalinSobol)
        {
            thrust::transform(device_input.begin(), device_input.end(),device_input.begin(),inverseCumulativeShawBrickman<float>());
        }

        TimeSoFar(totalStart,"inverse cumulativing done");


        bridgeMainDevice(n_vectors, 
            n_poweroftwo,
            device_input,
            dev_output);


        cudaThreadSynchronize();


        CUT_CHECK_ERR("after inverse\n");

        TimeSoFar(totalStart,"bridging done");

        // option data
        float spot =100.0f;
        float expiry =1.0f;
        float strike = 100.0f; // 80.0f+20.0f*(j % 3);
        float r = 0.05f;
        float d = 0.03f;
        float sigma = 0.1f +0.002f*j;
        int n_steps = n_dimensions;

        float logSpot = log(spot);
        float df = exp(-r*expiry);

        int samplingDates =n_dimensions;

        if (samplingDates > n_steps)
            throw("sampling dates must be less than n_steps");

        std::vector<float> logDrifts_vec(samplingDates);
        std::vector<float> logSds_vec(samplingDates);

        float dt =  expiry/samplingDates;
        // 1    2      3       4        5     6     7       8      9       10  11  12
        //float vols []= {0.19f , 0.19f, 0.19f, 0.19f, 0.19f, 0.19f, 0.19f, 0.19f, 0.19f, 0.19f,0.19f, 0.287923601f};

        for (int k=0; k < samplingDates; ++k)
        {          
            logSds_vec[k] = sigma*sqrt(dt);


            logDrifts_vec[k] = (r-d)*dt-0.5f*logSds_vec[k]*logSds_vec[k];

        }

        float strikeArithmetic = strike;
        float strikeGeometric = strike;


        std::cout << " enter  Asian\n";    

        AsianMainDevice(
            dev_output,
            n_dimensions, // must be greater than or equal to stepsPerPath
            n_vectors, 
            samplingDates,
            logDrifts_vec, 
            logSds_vec, 
            logSpot,
            df,
            strikeArithmetic,
            outputDataArithmetic,
            strikeGeometric,
            outputDataGeometric );


        TimeSoFar(totalStart,"paths done");

        float meanArith(thrust::reduce  	(  outputDataArithmetic.begin(),outputDataArithmetic.end(),0.0f));
        float meanGeometric(thrust::reduce  	(  outputDataGeometric.begin(),outputDataGeometric.end(),0.0f));


        std::cout << " exit  Asian\n";    




        int totalFinish = clock();

        float totalTime = (totalFinish-totalStart)/(static_cast<float>(CLOCKS_PER_SEC));

        std::cout << "total time: " << totalTime << std::endl;
        std::cout << "mean arirthmetic " << meanArith/n_vectors << "      meanGeometric " << meanGeometric/n_vectors << "\n";

        results[j]= meanArith/n_vectors -        meanGeometric/n_vectors   ;
        std::cout << "controlled value " <<        results[j]              << "\n";         


    }


    int loopsFinish = clock();

    float totalTimeLoops = (loopsFinish-totalLoopTime)/(static_cast<float>(CLOCKS_PER_SEC));

    std::cout << " average loop time " << totalTimeLoops/loops << "\n";

    for (int l=0; l < loops; ++l)
        std::cout << results[l] << "\n";

    cudaThreadExit();

    if (waitAtEnd) 
    {           
        char c;
        std::cin >> c;
    }

    return 0;
}


