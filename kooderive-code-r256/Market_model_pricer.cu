//
//
//                                                                                                  Market_model_pricer.cu
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

#include "Market_model_pricer.h"
#include <cutil.h>
#include "LMM_evolver_full.h"
#include "LMM_evolver_gpu.h"
#include "LMM_evolver_main.h"
#include "allocate_thrust.h"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include "cudaMacros.h"
#include <gold/MatrixFacade.h> 
#include <gold/Bridge_gold.h>
#include "cashFlowGeneration_product_gpu.h"
#include "utilities.h"
#include <gold/cashFlowDiscounting_gold.h>
#include "cashFlowDiscounting_main.h"

#include "cashFlowDiscounting_gpu.h"

float LMMPricerRoutine(const LMMPricerRoutineData& inputs,
                       int scrambleOffset,
                       int pathOffset, 
                       int DeviceToUse,
					   bool fermiArch,
					   bool useSharedForDiscounting
                       )
{

    if (inputs.firstCashFlowsTimes.size() != inputs.stepsForEvolution)
        throw("firstCashFlowsTimes is not of stepsForEvolution size. ");

    if (inputs.secondCashFlowsTimes.size() != inputs.stepsForEvolution)
        throw("secondCashFlowsTimes is not of stepsForEvolution size. ");

	if (fermiArch)
		throw("fermi optimizations not yet supported by LMMPricerRoutine");

    float result =-1.0;

    bool useTexturesForDiscounting = true;

  Timer t1;

    int tot_dimensions = intPower(2,inputs.powerOfTwoForVariates);

    if (DeviceToUse == -1)
        DeviceToUse = cutGetMaxGflopsDeviceId();

    cudaSetDevice(DeviceToUse);
    { // scope to force destruction of all interior objects at end

        thrust::device_vector<float> evolved_rates_device; // for output
        {
            thrust::device_vector<float> evolved_log_rates_device; // for output

            LMMEvolutionSemiDevRoutine( inputs.paths, 
                pathOffset, 
                inputs.rates, 
                inputs.factors, 
                inputs.stepsForEvolution, 
                inputs.powerOfTwoForVariates,
                inputs.scrambler_host, 
                scrambleOffset,
                inputs.pseudoRoots_host,
                inputs.fixedDrifts_host, 
                inputs.displacements_host,
                inputs.initial_rates_host, 
                inputs.initial_log_rates_host, 
                inputs.taus_host, 
                inputs.initial_drifts_host, 
                inputs.aliveIndices, 
                inputs.alive_host, 
                evolved_rates_device, // for output
                evolved_log_rates_device // for output
                );
        }

        // ok the rates have been evolved, now generate generic auxiliary data

        thrust::device_vector<float> taus_device(inputs.taus_host);
        thrust::device_vector<float>  discounts_device(inputs.paths*(inputs.rates+1)*inputs.stepsForEvolution);


        thrust::device_vector<float> numeraire_values_device(inputs.stepsForEvolution*inputs.paths);
        thrust::device_vector<float> select_forwards_device(inputs.stepsForEvolution*inputs.paths);

        thrust::device_vector<int> alive_device(   deviceVecFromStlVec(inputs.aliveIndices));
        bool doAllStepsAtOnce = true;

        discount_ratios_computation_main( evolved_rates_device, 
            taus_device, 
            inputs.aliveIndices, 
            alive_device,
            inputs.paths,
            inputs.stepsForEvolution, 
            inputs.rates, 
            discounts_device,  // for output 
            doAllStepsAtOnce
            );

     //   debugDumpVector(discounts_device,"discounts_device");                                                                                   






        spot_measure_numeraires_computation_gpu_main(   discounts_device,
            numeraire_values_device, //output
            inputs.paths,
            inputs.rates
            );


  //      debugDumpVector(numeraire_values_device,"numeraire_values_device");                   




        std::vector<int> forwardIndices(inputs.stepsForEvolution);

        for (int i=0; i < inputs.stepsForEvolution; ++i)
            forwardIndices[i] = i;


        forward_rate_extraction_gpu_main(   evolved_rates_device, 
            forwardIndices,                          
            inputs.paths,
            inputs.stepsForEvolution, 
            inputs.rates,   
            select_forwards_device                  
            );

 //       debugDumpVector(select_forwards_device,"select_forwards_device");        

        thrust::device_vector<float> genFlows1(inputs.stepsForEvolution*inputs.paths,0.0);
        thrust::device_vector<float> genFlows2(inputs.stepsForEvolution*inputs.paths,0.0);
        thrust::device_vector<float> auxData_device(inputs.auxData);

        float* rates1_devPtr =  thrust::raw_pointer_cast(&select_forwards_device[0]);
        float* rates2_devPtr = rates1_devPtr;
        float* rates3_devPtr = rates1_devPtr;

        // now call the product and get the cash-flows 

   //     debugDumpVector(auxData_device,"auxData_device");      
        float *aux_devPtr =  thrust::raw_pointer_cast(&auxData_device[0]);

        cashFlowGeneratorCallerTARN(thrust::raw_pointer_cast(&genFlows1[0]), 
            thrust::raw_pointer_cast(&genFlows2[0]), 
            aux_devPtr, 
            inputs.paths, 
            inputs.stepsForEvolution,
            rates1_devPtr, 
            rates2_devPtr, 
            rates3_devPtr, 
            thrust::raw_pointer_cast(&evolved_rates_device[0]), 
            thrust::raw_pointer_cast(&discounts_device[0])
            );
//
   //     debugDumpVector(genFlows1,"genFlows1");      
  //      debugDumpVector(genFlows2,"genFlows2");                                                                        

        // now discount the cash-flows 

        thrust::device_vector<float> discountedFlows1(inputs.paths*inputs.stepsForEvolution);
        thrust::device_vector<float> summedDiscountedFlows1(inputs.paths);
        thrust::device_vector<float> discountedFlows2(inputs.paths*inputs.stepsForEvolution);
        thrust::device_vector<float> summedDiscountedFlows2(inputs.paths);

        {
            std::vector<int> firstIndex1; 
            std::vector<int> secondIndex1;
            std::vector<float>  thetas1; 

            generateCashFlowIndicesAndWeights<float>( firstIndex1, 
                secondIndex1,
                thetas1, 
                inputs.rateTimes,
                inputs.firstCashFlowsTimes
                );

            thrust::device_vector<int> firstIndex1_dev(deviceVecFromStlVec(firstIndex1));

            thrust::device_vector<int> secondIndex1_dev(deviceVecFromStlVec(secondIndex1));
            thrust::device_vector<float> thetas1_dev(deviceVecFromStlVec(thetas1));

            cashFlowDiscounting_gpu_main(firstIndex1_dev, 
                secondIndex1_dev,
                thetas1_dev, 
                discounts_device, 
                genFlows1, 
                numeraire_values_device,
                inputs.paths, 
                inputs.stepsForEvolution,
                useTexturesForDiscounting,
				useSharedForDiscounting,
                discountedFlows1, // output
                summedDiscountedFlows1); // output

      //      debugDumpVector(discountedFlows1,"discountedFlows1");     
     //       debugDumpVector(summedDiscountedFlows1,"summedDiscountedFlows1");                                                

            std::vector<int> firstIndex2; 
            std::vector<int> secondIndex2;
            std::vector<float>  thetas2; 

            generateCashFlowIndicesAndWeights<float>( firstIndex2, 
                secondIndex2,
                thetas2, 
                inputs.rateTimes,
                inputs.secondCashFlowsTimes
                );

            thrust::device_vector<int> firstIndex2_dev(deviceVecFromStlVec(firstIndex2));

            thrust::device_vector<int> secondIndex2_dev(deviceVecFromStlVec(secondIndex2));
            thrust::device_vector<float> thetas2_dev(deviceVecFromStlVec(thetas2));

            cashFlowDiscounting_gpu_main(firstIndex2_dev, 
                secondIndex2_dev,
                thetas2_dev, 
                discounts_device, 
                genFlows2, 
                numeraire_values_device,
                inputs.paths, 
                inputs.stepsForEvolution, 
                useTexturesForDiscounting,
				useSharedForDiscounting,
                discountedFlows2, // output
                summedDiscountedFlows2); // output

       //     debugDumpVector(discountedFlows2,"discountedFlows2");     
      //      debugDumpVector(summedDiscountedFlows2,"summedDiscountedFlows2");                                                


        }

        // we have the discounted cash-flows now we need to reduce them

        float sum1= thrust::reduce(summedDiscountedFlows1.begin(),summedDiscountedFlows1.end());
        float sum2= thrust::reduce(summedDiscountedFlows2.begin(),summedDiscountedFlows2.end());

        result = (sum1+sum2)/inputs.paths;
    }
     std::cout << "time taken by complete market model pricer routine " << t1.timePassed() << "\n";
    cudaThreadExit();

    return result;
}

void LMMPricerRoutineMultiBatch(const LMMPricerRoutineData& inputs,
                                int pathOffsetPerBatch,
                                int scrambleOffsetPerBatch,
                                int baseScrambleOffset,
                                int basePathOffset, 
                                int batches,
                                int DeviceToUse,
                                float* outputs,
                                int outputsOffset,
                                int outputsStep,
                                bool doDiscounts,
                                bool newBridge,
								bool fermiArch,
								bool useSharedForDiscounting
                                )
{

    if (inputs.firstCashFlowsTimes.size() != inputs.stepsForEvolution)
        throw("firstCashFlowsTimes is not of stepsForEvolution size. ");

    if (inputs.secondCashFlowsTimes.size() != inputs.stepsForEvolution)
        throw("secondCashFlowsTimes is not of stepsForEvolution size. ");


    bool useTexturesForDiscounting = true;


	Timer t1;
	Timer tM;
   

    int tot_dimensions = intPower(2,inputs.powerOfTwoForVariates);
    int numberTotalVariates= inputs.paths*inputs.factors*tot_dimensions;
    int numberEvolvedRates = inputs.paths*inputs.rates*inputs.stepsForEvolution;

    if (DeviceToUse == -1)
        DeviceToUse = cutGetMaxGflopsDeviceId();

	std::cout << " Device to use " << DeviceToUse << "\n";
	std::cout << " Batches " << batches << "\n";
	std::cout << " paths  " << inputs.paths << "\n";


    cudaSetDevice(DeviceToUse);
    { // extra scope to force all device vectors and similar to be deallocated before
      // cudaThreadExit()

        {
            cudaEvent_t wakeGPU;
            cutilSafeCall( cudaEventCreate( &wakeGPU) );
        }

        //    cudaThreadSynchronize();

        std::vector<int> firstIndex1; 
        std::vector<int> secondIndex1;
        std::vector<float>  thetas1; 

        generateCashFlowIndicesAndWeights<float>( firstIndex1, 
            secondIndex1,
            thetas1, 
            inputs.rateTimes,
            inputs.firstCashFlowsTimes
            );
        std::vector<int> firstIndex2; 
        std::vector<int> secondIndex2;
        std::vector<float>  thetas2; 

        generateCashFlowIndicesAndWeights<float>( firstIndex2, 
            secondIndex2,
            thetas2, 
            inputs.rateTimes,
            inputs.secondCashFlowsTimes
            );


        //  cudaThreadSynchronize();


		Timer tS;


        allocate_device_thrust_pointer<float> evolved_log_alloc(numberEvolvedRates);
        float* evolved_log_rates_global = evolved_log_alloc.getGlobal();

        allocate_device_thrust_pointer<float> evolved_alloc(numberEvolvedRates);
        float* evolved_rates_global =evolved_alloc.getGlobal();

        allocate_device_thrust_pointer<float> discounts_alloc(inputs.paths*(inputs.rates+1)*inputs.stepsForEvolution);
        float* discounts_global = discounts_alloc.getGlobal();

        allocate_device_thrust_pointer<float>  numeraire_values_alloc(inputs.stepsForEvolution*inputs.paths); 
        float* numeraire_values_global = numeraire_values_alloc.getGlobal();

        allocate_device_thrust_pointer<float> correlatedVariates_alloc(numberEvolvedRates);
        float* correlatedVariates_global = correlatedVariates_alloc.getGlobal();

        allocate_device_thrust_pointer<float> select_forwards_alloc(inputs.stepsForEvolution*inputs.paths);
        float* select_forwards_global =  select_forwards_alloc.getGlobal();


        allocate_device_thrust_pointer<float> discountedFlows1_alloc(inputs.paths*inputs.stepsForEvolution);
        float* discountedFlows1_global =  discountedFlows1_alloc.getGlobal();

        allocate_device_thrust_pointer<float> discountedFlows2_alloc(inputs.paths*inputs.stepsForEvolution);
        float* discountedFlows2_global =  discountedFlows2_alloc.getGlobal();


        allocate_device_thrust_pointer<float> e_buffer_alloc(inputs.paths*inputs.factors);
        float* e_buffer_global =  e_buffer_alloc.getGlobal();


        allocate_device_thrust_pointer<float> e_buffer_pred_alloc(inputs.paths*inputs.factors);
        float* e_buffer_pred_global =  e_buffer_pred_alloc.getGlobal();



        allocate_device_thrust_pointer<float> genFlows1_alloc(inputs.stepsForEvolution*inputs.paths);
        float* genFlows1_global =  genFlows1_alloc.getGlobal();  

        allocate_device_thrust_pointer<float> genFlows2_alloc(inputs.stepsForEvolution*inputs.paths);
        float* genFlows2_global =  genFlows2_alloc.getGlobal();  

        allocate_device_thrust_pointer<float> summedDiscountedFlows1_alloc(inputs.paths);
        float* summedDiscountedFlows1_global =summedDiscountedFlows1_alloc.getGlobal();  

        allocate_device_thrust_pointer<float> summedDiscountedFlows2_alloc(inputs.paths);
        float* summedDiscountedFlows2_global =summedDiscountedFlows2_alloc.getGlobal();  



        thrust::device_vector<unsigned int> scrambler_device(inputs.factors*tot_dimensions); 

        thrust::device_vector<unsigned int> SobolInts_buffer_device(numberTotalVariates);
        unsigned int* SobolInts_buffer_global =  thrust::raw_pointer_cast(&SobolInts_buffer_device[0]);  

        thrust::device_vector<float> quasiRandoms_buffer_device(numberTotalVariates);       
        float* quasiRandoms_buffer_global =  thrust::raw_pointer_cast(&quasiRandoms_buffer_device[0]);  

        thrust::device_vector<float> bridgeVariates_device(numberTotalVariates);
        float* bridgeVariates_global =  thrust::raw_pointer_cast(&bridgeVariates_device[0]);  


        // now pass input data to GPU    
        thrust::device_vector<float> taus_device(inputs.taus_host);
        float* taus_global = thrust::raw_pointer_cast(&taus_device[0]);  

        thrust::device_vector<float> auxData_device(inputs.auxData);
        float* auxData_global =  thrust::raw_pointer_cast(&auxData_device[0]);  

        thrust::device_vector<int> firstIndex1_dev(deviceVecFromStlVec(firstIndex1));
        int* firstIndex1_global =  thrust::raw_pointer_cast(&firstIndex1_dev[0]);  

        thrust::device_vector<int> secondIndex1_dev(deviceVecFromStlVec(secondIndex1));
        int* secondIndex1_global =  thrust::raw_pointer_cast(&secondIndex1_dev[0]);  

        thrust::device_vector<float> thetas1_dev(deviceVecFromStlVec(thetas1));
        float* thetas1_global =  thrust::raw_pointer_cast(&thetas1_dev[0]);  

        thrust::device_vector<int> firstIndex2_dev(deviceVecFromStlVec(firstIndex2));
        int* firstIndex2_global =  thrust::raw_pointer_cast(&firstIndex2_dev[0]);  

        thrust::device_vector<int> secondIndex2_dev(deviceVecFromStlVec(secondIndex2));
        int* secondIndex2_global =  thrust::raw_pointer_cast(&secondIndex2_dev[0]);  

        thrust::device_vector<float> thetas2_dev(deviceVecFromStlVec(thetas2));
        float* thetas2_global =  thrust::raw_pointer_cast(&thetas2_dev[0]);  


        thrust::device_vector<float> pseudoRoots_device(inputs.pseudoRoots_host);   
        thrust::device_vector<float> fixedDrifts_device(inputs.fixedDrifts_host); 
        thrust::device_vector<float> displacements_device(inputs.displacements_host);
        thrust::device_vector<float> initial_rates_device(inputs.initial_rates_host); 
        thrust::device_vector<float> initial_log_rates_device(inputs.initial_log_rates_host); 
        thrust::device_vector<float> initial_drifts_device(inputs.initial_drifts_host);

        thrust::device_vector<int> alive_device(inputs.alive_host);

        std::vector<int> forwardIndices(inputs.stepsForEvolution);

        for (int j=0; j < inputs.stepsForEvolution; ++j)
            forwardIndices[j] = j;

        //     cudaThreadSynchronize();

       
        double timeM =tM.timePassed();
        std::cout << "time taken on memory allocation , " << timeM << "\n";

        double timeS = tS.timePassed();
        std::cout << "time taken memory routine excluding wake up " << timeS << "\n";


        int i = outputsOffset;

		std::cout << "  Device: " << DeviceToUse 
					<< "; about to do part of" << batches << " batches. The offset is " << outputsOffset << "\n";

        while  (i < batches)
        {  
	
            Timer tB;


            int pathOffset = basePathOffset+pathOffsetPerBatch*i;
            int scrambleOffset = baseScrambleOffset+scrambleOffsetPerBatch*i;

            thrust::copy(inputs.scrambler_host.begin()+scrambleOffset,inputs.scrambler_host.begin()+scrambleOffset+scrambler_device.size(),scrambler_device.begin() );

            bool singleKernel = true;

	   
			std::cout << "  Device: " << DeviceToUse << "; about to do batch" << i << " . The offset is " << pathOffset << "\n";


            if (singleKernel)
                LMMEvolutionRoutineRawSingleKernel(inputs.paths, 
                pathOffset, 
                inputs.rates, 
                inputs.factors, 
                inputs.stepsForEvolution, 
                inputs.powerOfTwoForVariates,
                scrambler_device, 
                pseudoRoots_device,
                fixedDrifts_device, 
                displacements_device,
                initial_rates_device, 
                initial_log_rates_device, 
                taus_device, 
                initial_drifts_device, 
                inputs.aliveIndices, 
                alive_device, 
                SobolInts_buffer_device, 
                quasiRandoms_buffer_device, 
                bridgeVariates_device, 
                correlatedVariates_global, 
                e_buffer_global,
                e_buffer_pred_global,
                evolved_rates_global,
                evolved_log_rates_global,  // for output 
                discounts_global,
                inputs.useSharedWhereChoice,
                doDiscounts,
                newBridge,
				fermiArch,
				0 // threads
                ); 
            else
                LMMEvolutionRoutineRaw(inputs.paths, 
                pathOffset, 
                inputs.rates, 
                inputs.factors, 
                inputs.stepsForEvolution, 
                inputs.powerOfTwoForVariates,
                scrambler_device, 
                pseudoRoots_device,
                fixedDrifts_device, 
                displacements_device,
                initial_rates_device, 
                initial_log_rates_device, 
                taus_device, 
                initial_drifts_device, 
                inputs.aliveIndices, 
                alive_device, 
                SobolInts_buffer_device, 
                quasiRandoms_buffer_device, 
                bridgeVariates_device, 
                correlatedVariates_global, 
                e_buffer_global,
                e_buffer_pred_global,
                evolved_rates_global, // for output
                evolved_log_rates_global  // for output 
				);


            // ok the rates have been evolved, now generate generic auxiliary data

            if (!doDiscounts)
            {
                bool allStepsAtOnce = true;

                discount_ratios_computation_gpu(    evolved_rates_global, 
                    taus_global, 
                    inputs.aliveIndices, 
                    thrust::raw_pointer_cast(&alive_device[0]),
                    inputs.paths,
                    inputs.stepsForEvolution, 
                    inputs.rates, 
                    discounts_global, // for output 
                    allStepsAtOnce);

            }                                                          






            spot_measure_numeraires_computation_gpu(   discounts_global,
                numeraire_values_global, //output
                inputs.paths,
                inputs.rates
                );



            forward_rate_extraction_gpu(   evolved_rates_global, 
                forwardIndices,                          
                inputs.paths,
                inputs.stepsForEvolution, 
                inputs.rates,   
                select_forwards_global                 
                );



            float* rates1_global =  select_forwards_global;
            float* rates2_global = select_forwards_global;
            float* rates3_global = select_forwards_global;

            // now call the product and get the cash-flows 


            cashFlowGeneratorCallerTARN(genFlows1_global, 
                genFlows2_global, 
                auxData_global, 
                inputs.paths, 
                inputs.stepsForEvolution,
                rates1_global, 
                rates2_global, 
                rates3_global, 
                evolved_rates_global, 
                discounts_global
                );
            // now discount the cash-flows 


            cashFlowDiscounting_gpu(firstIndex1_global, 
                secondIndex1_global,
                thetas1_global, 
                discounts_global, 
                genFlows1_global, 
                numeraire_values_global,
                inputs.paths, 
                inputs.stepsForEvolution, 
                useTexturesForDiscounting,
				useSharedForDiscounting,
                discountedFlows1_global, // output
                summedDiscountedFlows1_global); // output


            cashFlowDiscounting_gpu(firstIndex2_global, 
                secondIndex2_global,
                thetas2_global, 
                discounts_global, 
                genFlows2_global, 
                numeraire_values_global,
                inputs.paths, 
                inputs.stepsForEvolution, 
                useTexturesForDiscounting,
			    useSharedForDiscounting,
                discountedFlows2_global, // output
                summedDiscountedFlows2_global); // output



            // we have the discounted cash-flows now we need to reduce them

            float sum1= thrust::reduce(summedDiscountedFlows1_alloc.getDevicePtr(),summedDiscountedFlows1_alloc.getEndPtr());
            float sum2= thrust::reduce(summedDiscountedFlows2_alloc.getDevicePtr(),summedDiscountedFlows2_alloc.getEndPtr());

            float result = (sum1+sum2)/inputs.paths;
            outputs[i] = result;


            double timeB = tB.timePassed();

            std:: cout << "time for batch, " << i << ",  is ," << timeB << "\n"; 

            i+= outputsStep;
        }



      
        double time = t1.timePassed();
        std::cout << "time taken by batched complete market model pricer routine " << time << "\n";
    }
    cudaThreadExit();

    return;
}
