//
//
//                                                                                                                                          MultiD_Path_Gen_Test.cu
//
//
// (c) Mark Joshi 2010,2014
// This code is released under the GNU public licence version 3

#include "MultiD_Path_Gen_BS_Test.h"
#include <gold/MultiD_Path_Gen_BS_gold.h>
#include "multid_path_gen_BS_main.h"
#include <gold/weighted_average_gold.h>
#include <weighted_average_main.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cutil_inline.h>
#include <cutil.h>

#include <cuda_runtime.h>
#include "InverseCumulatives.h"
#include <gold/Bridge_gold.h>
#include "sobol.h"
#include "Bridge.h"
#include "Correlation_main.h"
#include "drift_add_main.h"
#include "cudaMacros.h"
#include <gold/Correlation_gold.h>
#include <ComparisonCheck.h>
#include <gold/cashFlowGeneration_equity_product_oned_gold.h>
#include <cash_flow_product_equity_one_dim_main.h>

#include "Utilities.h"
// compare GPU against CPU

namespace
{
    double tolerance = 1E-4;

}


int MultiDBSTestRoutine(bool verbose, bool useTextures, DeviceChooser& chooser)
{


    int path_offset =0;
    int result =3;
    {
        if (verbose)
        {
            std::cout << "\n\nTesting correlation routine ";
            if (!useTextures)
                std::cout << " not ";
            std::cout << " using textures.\n";
        }
        bool extraVerbose = false;

        cudaSetDevice(chooser.WhichDevice());


        int n_vectors =  intPower(2,19);
        int n_poweroftwo =5;
        int factors = 10;
        int n_steps = intPower(2,n_poweroftwo);
        int tot_dimensions = n_steps*factors;
        int N= n_vectors*tot_dimensions; 
        int number_rates = 10;


 
        		size_t outN = n_vectors*n_steps*number_rates;
            
         size_t estimatedFloatsRequired = 4*(2*outN+2*N);
   
         size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

 //  bool change=false;
  
       ConfigCheckForGPU checker;

       float excessMem=2.0f;

       while (excessMem >0.0f)
       {
           checker.checkGlobalMem( estimatedMem, excessMem );

           if (excessMem >0.0f)
           {
               n_vectors/=2;
               N= n_vectors*tot_dimensions;
               outN = n_vectors*n_steps*number_rates;
  //             change =true;
               estimatedFloatsRequired = 4*(2*outN+2*N);
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
     
           }
       }

        std::vector<float> A(number_rates*factors);

        for (int i=0; i < static_cast<int>(A.size()); ++i)
            A[i] = static_cast<float>(0.01f*(i+1.0f)/A.size());



        thrust::host_vector<float> A_host(A.begin(),A.end());
        thrust::device_vector<float> A_device(A_host);

        std::vector<int> alive(n_steps);
        std::fill(alive.begin(), alive.end(),0);

        thrust::host_vector<int> alive_host(alive.begin(),alive.end());
        thrust::device_vector<int> alive_device(alive_host);



        std::vector<float> drifts(n_steps*number_rates);
        for (int i=0; i < static_cast<int>(drifts.size()); ++i)
            drifts[i] = static_cast<float>(i/(drifts.size()*100.0f));

        thrust::host_vector<float> drifts_host(drifts.begin(),drifts.end());
        thrust::device_vector<float> drifts_device(drifts_host);


        std::vector<float> logRates_vec(number_rates);
        for (int i=0; i < number_rates; ++i)
            logRates_vec[i] = log(1.0f+i*0.05f);

        thrust::host_vector<float> logRates_host(logRates_vec.begin(),logRates_vec.end());
        thrust::device_vector<float> logRates_device(logRates_host);






        cudaThreadSynchronize();

        thrust::device_vector<float> dev_output(N);    
        thrust::device_vector<float> device_input(N);

        bool doNormalInSobol=true;
        SobDevice( n_vectors, tot_dimensions,path_offset, device_input,doNormalInSobol);




        cudaThreadSynchronize();


        if (extraVerbose)
        {
            thrust::host_vector<float> bridgedNumbersHost(N);   
            bridgedNumbersHost = device_input;

            for (int i=0; i < N; ++i)
                std::cout << i << "," << bridgedNumbersHost[i]  << "\n";

        }


        BrownianBridgeMultiDim<float>::ordering allocator( BrownianBridgeMultiDim<float>::triangular);


        MultiDBridge(n_vectors, 
            n_poweroftwo,
            factors,
            device_input, 
            dev_output,
            allocator,
            useTextures)
            ;



        thrust::host_vector<float> bridgedNumbersHost(N);   
        bridgedNumbersHost = dev_output;
        std::vector<float> bridgedNumbers_vec(bridgedNumbersHost.begin(), bridgedNumbersHost.end());

        thrust::device_vector<float> dev_correlated_rates(outN);
        thrust::device_vector<float> outputPaths_BSGenerator_device(outN);


        if (extraVerbose)
        {
            std::cout << "logRates_device " << "\n";

            for (int i=0; i < number_rates; ++i)
                std::cout << logRates_device[i] << ",";

            std::cout << "\n";
        }

      Timer h1;

        correlated_paths_main_multistep_device(dev_output, // randon numbers
            dev_correlated_rates, // correlated rate increments 
            A_device, // correlator 
            alive_device,
            0,
            factors, 
            number_rates,
            n_vectors,
            n_steps);


        drift_add_device(drifts_device,  // matrix to add
            dev_correlated_rates, // data to add to 
            n_steps, 
            number_rates ,
            n_vectors,
            useTextures); 

        multi_dim_BS_path_generator_device( logRates_device,
            dev_correlated_rates,
            outputPaths_BSGenerator_device, 
            n_vectors,
            number_rates,
            n_steps);


        double time = h1.timePassed();
        std::cout << " time taken for all steps of path generation."    << time << std::endl;


        std::vector<float> outputPaths_vec(n_vectors*number_rates*n_steps);
        std::vector<float> correlatedNumbersVec(n_vectors*number_rates*n_steps);



        int t0=clock();
        for (int s=0; s < n_steps; ++s)
            correlated_paths_gold(bridgedNumbers_vec, // randon numbers
            correlatedNumbersVec, // correlated rate increments 
            A, // correlator 
            factors, 
            number_rates,
            n_vectors,
            s);


        multi_dim_BS_path_generator_nonlog_gold(logRates_vec,
            correlatedNumbersVec,
            drifts,
            outputPaths_vec, 
            n_vectors,
            number_rates,
            n_steps);

        int t1=clock();

        float time2 = (t1-t0+0.0f)/CLOCKS_PER_SEC;

        std::cout << " time taken for CPU correlation " << time2 << "\n";

        std::cout << " speed up ratio " << time2/time << "\n";

        // retrieve the final paths from gpu 
        thrust::host_vector<float> outPaths_gpu_host(outputPaths_BSGenerator_device);
        std::vector<float> outPaths_gpu_vec(outPaths_gpu_host.begin(), outPaths_gpu_host.end());


        double err1=0.0;
        double err2 = 0.0;



        for (int p = 0; p < n_vectors; ++p)
            for (int s=0; s < n_steps; ++s)
            {
                if (extraVerbose)
                    std::cout << p << "," << s ;

                for (int r=0; r < number_rates; ++r)     
                {
                    int i = p +    n_vectors*(r+ s*number_rates);
                    if (extraVerbose)
                        std::cout <<  "," << outPaths_gpu_vec[i] ;
                    double erri =2* (outPaths_gpu_vec[i] - outputPaths_vec[i])/(outPaths_gpu_vec[i] + outputPaths_vec[i]);
                    err1 += fabs(erri);
                    err2 += fabs(erri*erri);
                }

                if (extraVerbose)
                {
                    for (int r=0; r < number_rates; ++r)     
                    {
                        int i = p +    n_vectors*(r+ s*number_rates);
                        std::cout <<  "," << outputPaths_vec[i] ;
                    }

                    std::cout << "\n";
                }
            }


            double L1err = err1/(outN-number_rates*n_steps);

            double L2err = sqrt(err2/(outN-number_rates*n_steps));


            if (verbose)
            {
                std::cout << "L1  error " << L1err << "\n";
                std::cout << "L2  error " << L2err << "\n";
            }

            if (L1err > tolerance)
            {
                std::cout << " multiD path gen  test failed";
            }
            else
                --result;


            // now test basket code  
            std::vector<float> weights_vec(number_rates);

            for (size_t i=0; i < weights_vec.size(); ++i)
                weights_vec[i] = (i+0.5f)/(number_rates*number_rates);


            thrust::host_vector<float> weights_host(weights_vec.begin(),weights_vec.end());
            thrust::device_vector<float> weights_dev(weights_host);

            thrust::device_vector<float> outputPaths_dev( n_vectors*n_steps);

            CubeConstFacade<float> input_w_Paths_cube(&outputPaths_vec[0],n_steps,number_rates,n_vectors);


            std::vector<float> output_w_gold(n_vectors*n_steps);

            MatrixFacade<float> output_w_mat(&output_w_gold[0],n_steps,n_vectors);

            std::cout << " basket weightings test.";

            Timer h3;

            basketWeightings_gold(input_w_Paths_cube, 
                output_w_mat, 
                weights_vec,
                n_vectors,
                number_rates,
                n_steps);


            double time3 = h3.timePassed();
            std::cout << " time taken for all averaging on CPU: "    << time3 << std::endl;


           Timer h4;


            basketWeightings_main(outputPaths_BSGenerator_device, 
                outputPaths_dev, 
                weights_dev,
                n_vectors,
                number_rates,
                n_steps);

            double time4 = h4.timePassed();
            std::cout << " time taken for all averaging on GPU: "    << time4 << std::endl;

            std::cout << " speed up ratio " << time3/time4 << std::endl;

            int errors = numberMismatches(output_w_mat,outputPaths_dev, tolerance, false);

            if (errors >0)
            {
                std::cout << " test failed with " << errors << " mismatches.\n";

            }
            else
            {

                std::cout << " test passed with " << errors << " mismatches.\n";
                --result;
            }

// now let's get some prices!

            std::cout << "\n ratchet payoff test \n";

            std::vector<float> genFlows_cpu(n_vectors);
            thrust::device_vector<float>  genFlows_gpu(n_vectors);

            float strike=1.0;
            thrust::device_vector<float> auxData_dev(1);
            auxData_dev[0] =strike;

            RatchetPayoffs_gold<float>( genFlows_cpu,
                      strike,
                      n_steps,
                      n_vectors,
                      output_w_mat);


            payoffGeneratorCallerRatchetPut_main(genFlows_gpu,  
                                          auxData_dev, 
                                          n_vectors, 
                                          n_steps,
                                          outputPaths_dev
                                          );

            
            int errors2 = numberMismatches(genFlows_cpu, genFlows_gpu,  tolerance, false);

 
            if (errors2 >0)
            {
                std::cout << " test failed with " << errors2 << " mismatches.\n";

            }
            else
            {

                std::cout << " test passed with " << errors2 << " mismatches.\n";
                --result;
            }


    }

    cudaThreadExit();


    return result;
}
