//
//                                               TARNCUDAPricer.cpp
//
//
//
#include <math.h>

#include "TARNCUDAPricer.h"
#include "marketmodelevolvercuda.h"
#include "Market_model_pricer.h"
#include "Utilities.h"
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <cutil_inline.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include "multithreading.h"



void TARNPricerRoutine(const MarketModelEvolverLMMPC_CUDA& evolver,
                                            int batches,
                                            const std::vector<float>& auxData,
                                            const std::vector<float>& firstCashFlowsTimes, 
                                            const std::vector<float>& secondCashFlowsTimes,
                                            std::vector<float>& batchValues_vec
                        )
{

      Timer h1;

     thrust::host_vector<unsigned int> scrambler_host(  evolver.GetmarketModel()->numberOfFactors()*evolver.GetmarketModel()->numberOfSteps()*batches,0);

     MersenneTwisterUniformRng rng(evolver.GetScramblingRNG());
     if (evolver.Getscramble())
     {  
                for (int j=0; j < static_cast<int>(scrambler_host.size()); ++j)
                      scrambler_host[j] =static_cast<unsigned int>(rng.nextInt32());
      }    

       thrust::host_vector<float> auxData_host(  hostVecFromStlVec<float>(auxData));

       thrust::host_vector<float> rateTimes_host(  hostVecCastStlVec<float,double>(evolver.GetmarketModel()->evolution().rateTimes()));
       std::vector<float> rateTimes(stlVecFromHostVec<float>(rateTimes_host));

      int scrambler_offset =0;
      int scrambleOffsetPerBatch=0;

       batchValues_vec.resize(batches);

       int evOffset = evolver.GetBatchSize();

       if (evolver.Getscramble())
       {
           evOffset = 0;
           scrambleOffsetPerBatch =  evolver.GetmarketModel()->numberOfFactors()*evolver.GetmarketModel()->numberOfSteps();
       }

        LMMPricerRoutineData inputs = {evolver.GetBatchSize(),
                                      evolver.GetmarketModel()->numberOfRates(), 
                                      evolver.GetmarketModel()->numberOfFactors(), 
                                      evolver.GetmarketModel()->numberOfSteps(), 
                                      evolver.GetpowerOfTwoForVariates(),
                                      scrambler_host, 
                                      evolver.GetpseudoRoots_host(),
                                      evolver.GetfixedDrifts_host(), 
                                      evolver.Getdisplacements_host(),
                                      evolver.Getinitial_rates_host(), 
                                      evolver.Getinitial_log_rates_host(), 
                                      evolver.Gettaus_host(), 
                                      evolver.Getinitial_drifts_host(), 
                                      evolver.Getalive(),
                                      evolver.Getalive_host(),
                                      auxData_host,
                                      firstCashFlowsTimes, 
                                      secondCashFlowsTimes,
                                      rateTimes};

     for (int i=0; i < batches;++i)
     {
       
         int scrambleOffsetThisBatch = scrambleOffsetPerBatch*i;
         int pathOffsetThisBatch = evolver.GetpathOffset()+evOffset*i;

         batchValues_vec[i] = 
                    LMMPricerRoutine(inputs,scrambleOffsetThisBatch,pathOffsetThisBatch);
                             
    

     }
   
         double time = h1.timePassed();
         std::cout << "time taken to do all batches " << time << "\n";


}

struct LMMPricerThreadData
{
    const LMMPricerRoutineData* generalData;
    int totalNumberOfPathsBatches;
    int scrambleOffsetPerBatch;
    int pathOffsetPerBatch;
    int initialScrambleOffset;
    int initialPathOffset;

    int stepIndex;
    int startIndex;

    bool doDiscounts;
    bool newBridge;

    float* outputs;
    
    int DeviceToUse;

	bool fermiArch;

	bool useSharedForDiscounting;

};

void TARNPricerRoutineOneThreadMultiBatch(LMMPricerThreadData* plan)
{
         LMMPricerRoutineMultiBatch(*plan->generalData,
                                           plan->pathOffsetPerBatch,
                                           plan->scrambleOffsetPerBatch,
                                           plan->initialScrambleOffset,
                                           plan->initialPathOffset, 
                                           plan->totalNumberOfPathsBatches,
                                           plan->DeviceToUse,
                                           plan->outputs,
                                           plan->startIndex,
                                           plan->stepIndex,
                                           plan->doDiscounts,
                                           plan->newBridge,
										   plan->fermiArch,
										   plan->useSharedForDiscounting
                                                        );
}


void TARNPricerRoutineOneThread(LMMPricerThreadData* plan)
{
    int batchNumber = plan->startIndex;

    while ( batchNumber < plan->totalNumberOfPathsBatches)
    {
         int scrambleOffsetThisBatch = plan->initialScrambleOffset+plan->scrambleOffsetPerBatch*batchNumber;
         int pathOffsetThisBatch = plan->initialPathOffset+plan->pathOffsetPerBatch*batchNumber;
        
         float result = LMMPricerRoutine(*plan->generalData,scrambleOffsetThisBatch,pathOffsetThisBatch,plan->DeviceToUse);
   
        plan->outputs[batchNumber] = result;
        batchNumber+= plan->stepIndex;
    }
}

void TARNPricerRoutineMultiThread(const MarketModelEvolverLMMPC_CUDA& evolver,
                                            int batches,
                                            const std::vector<float>& auxData,
                                            const std::vector<float>& firstCashFlowsTimes, 
                                            const std::vector<float>& secondCashFlowsTimes,
                                            int numberOfThreadsToUse, 
                                            std::vector<float>& batchValues_vec,
                                            bool multiBatch,
                                            bool useShared,
                                            bool mergeDiscounts,
                                            bool newBridge,
                                            std::vector<int> deviceIndex,
											bool fermiArch,
											bool sharedForDiscounting
                        )
{

    int numberGPUs;
    cutilSafeCall( cudaGetDeviceCount(&numberGPUs) );

    if (numberGPUs<numberOfThreadsToUse)
    {
        numberOfThreadsToUse = numberGPUs;
        std::cout << "\nUsing " << numberGPUs << " GPus since that's all that's available!\n";
    }
    else 
         std::cout << "\nUsing " << numberOfThreadsToUse << " GPus as requested!\n";

    if (deviceIndex.size() ==0)
    {
        deviceIndex.resize(numberOfThreadsToUse);
        for (Size i=0; i < deviceIndex.size(); ++i)
            deviceIndex[i] = i;
    }

    while (static_cast<int>(deviceIndex.size()) < numberOfThreadsToUse)
    {
        deviceIndex.push_back(deviceIndex.size());
        std::cout << "extending device index vector since too short.";
    }

   
     thrust::host_vector<unsigned int> scrambler_host(  evolver.GetmarketModel()->numberOfFactors()*evolver.GetmarketModel()->numberOfSteps()*batches,0);

     MersenneTwisterUniformRng rng(evolver.GetScramblingRNG());
     if (evolver.Getscramble())
     {  
                for (int j=0; j < static_cast<int>(scrambler_host.size()); ++j)
                {
                    int number = rng.nextInt32();
                    int number2 = number | 7;
                      scrambler_host[j] =static_cast<unsigned int>(number2);
                }
      }    

       thrust::host_vector<float> auxData_host(  hostVecFromStlVec<float>(auxData));

       thrust::host_vector<float> rateTimes_host(  hostVecCastStlVec<float,double>(evolver.GetmarketModel()->evolution().rateTimes()));
       std::vector<float> rateTimes(stlVecFromHostVec<float>(rateTimes_host));

      int scrambler_offset =0;
      int scrambleOffsetPerBatch=0;

       batchValues_vec.resize(batches);

       int evOffset = evolver.GetBatchSize();

       if (evolver.Getscramble())
       {
           evOffset = 0;
           scrambleOffsetPerBatch =  evolver.GetmarketModel()->numberOfFactors()*evolver.GetmarketModel()->numberOfSteps();
       }

        LMMPricerRoutineData inputs = {evolver.GetBatchSize(),
                                      evolver.GetmarketModel()->numberOfRates(), 
                                      evolver.GetmarketModel()->numberOfFactors(), 
                                      evolver.GetmarketModel()->numberOfSteps(), 
                                      evolver.GetpowerOfTwoForVariates(),
                                      scrambler_host, 
                                      evolver.GetpseudoRoots_host(),
                                      evolver.GetfixedDrifts_host(), 
                                      evolver.Getdisplacements_host(),
                                      evolver.Getinitial_rates_host(), 
                                      evolver.Getinitial_log_rates_host(), 
                                      evolver.Gettaus_host(), 
                                      evolver.Getinitial_drifts_host(), 
                                      evolver.Getalive(),
                                      evolver.Getalive_host(),
                                      auxData_host,
                                      firstCashFlowsTimes, 
                                      secondCashFlowsTimes,
                                      rateTimes,
                                      useShared};

        std::vector<LMMPricerThreadData> threadData(numberOfThreadsToUse);

        for (int i=0; i < numberOfThreadsToUse; ++i)
        {
            LMMPricerThreadData thisOne = { &inputs, batches, 
                scrambleOffsetPerBatch, evOffset,0, 
				evolver.GetpathOffset(), numberOfThreadsToUse,deviceIndex[i],mergeDiscounts,newBridge,&batchValues_vec[0],i, fermiArch,sharedForDiscounting};

            threadData[i] = thisOne;
        }
        
       Timer h1;

        const int MAX_GPU_COUNT = 8;
        CUTThread threadID[MAX_GPU_COUNT];
        int t0=clock();
        std::vector<int> times(numberOfThreadsToUse+1);
         for (int i =0; i < numberOfThreadsToUse; ++i)
         {
             if (multiBatch)
                    threadID[i] = cutStartThread((CUT_THREADROUTINE)TARNPricerRoutineOneThreadMultiBatch, &threadData[i]);
             else
                    threadID[i] = cutStartThread((CUT_THREADROUTINE)TARNPricerRoutineOneThread, &threadData[i]);
            times[i] = clock();      
         }

          cutWaitForThreads(threadID, numberOfThreadsToUse);

          times[numberOfThreadsToUse] =clock();

        
         double time = h1.timePassed();
         std::cout << "time taken to do all batches " << time << "\n";
         for (int i=0; i < static_cast<int>(times.size()); ++i)
                 std::cout << times[i]-t0<< ",";

         std::cout << "\n";


}
