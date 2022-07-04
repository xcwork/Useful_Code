//
//
//                                              ratchet.cu
//
//
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3


#include <multid_path_gen_BS_main.h>
#include <scramble_main.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <inverseCumulatives.h>
#include <gold/Bridge_gold.h>
#include <sobol.h>
#include <Bridge.h>
#include <correlate_drift_main.h>
#include <Correlate_drift_gpu.h>
#include "cudaMacros.h"
#include <gold/MatrixFacade.h> 

#include <Utilities.h>
#include <brownian_bridge.h>
#include <gold/Bridge_allocator_gold.h>
#include <cash_flow_product_equity_one_dim_main.h>
#include <weighted_average_main.h>
#include <gold/math/pseudoSquareRoot.h>
#include <output_device.h>

namespace
{

    void chooseDevice()
    {
        cudaSetDevice(0);
    }
}

void MultiDBSDoPaths(int paths, 
                     int pathOffset, 
                     int stocks, 
                     int factors, 
                     int stepsForEvolution, 
                     int powerOfTwoForVariates,
                     thrust::device_vector<unsigned int>& scrambler_device, 
                     thrust::device_vector<float>& pseudoRoots_device,
                     thrust::device_vector<float>& fixedDrifts_device, 
                     thrust::device_vector<float>& initial_stocks_device, 
                     thrust::device_vector<float>& initial_log_stocks_device,  
                     thrust::device_vector<int>& alive_device,
                     // buffers
                     thrust::device_vector<unsigned int>& SobolInts_buffer_device, 
                     thrust::device_vector<float>& quasiRandoms_buffer_device, 
                     thrust::device_vector<float>& bridgeVariates_device, 
                     thrust::device_vector<float>& correlatedVariates_device, 
                     thrust::device_vector<float>& evolved_stocks_device
                     // for output
                     // ,thrust::device_vector<float>& evolved_log_stocks_device  // for output 
                     )
{


    int tot_dimensions = intPower(2,powerOfTwoForVariates);

    // does some initializing and calls sobolGPUInts

    SobDeviceInts( paths, tot_dimensions*factors, pathOffset,SobolInts_buffer_device);

    cudaThreadSynchronize();

    // forwards to  scramble_gpu      
    // scrambling is in place
    scramble_main(SobolInts_buffer_device, // random numbers
        scrambler_device,
        tot_dimensions,
        paths);

    cudaThreadSynchronize();

    inverseCumulativeShawBrickmanUnsignedInt<float> inv;


    thrust::transform(SobolInts_buffer_device.begin(), SobolInts_buffer_device.end(),quasiRandoms_buffer_device.begin(),inv);

    cudaThreadSynchronize();





    // calls brownianBridgeMultiGPUReorder, and bridgeMainDevice
    std::vector<int> indices(factors*stepsForEvolution);
    bridge_allocate_diagonal( indices,factors, stepsForEvolution);
    brownian_bridge bb(stepsForEvolution, indices);
    bb.transform( quasiRandoms_buffer_device, 
        bridgeVariates_device,
        paths, 
        stepsForEvolution, 
        factors);

  
    cudaThreadSynchronize();

    // forwards to correlated_drift_paths_GPU
    correlated_drift_paths_device( bridgeVariates_device,
        correlatedVariates_device,
        pseudoRoots_device, // correlator 
        alive_device,
        fixedDrifts_device,
        factors*stocks,
        factors, 
        stocks,
        paths,
        stepsForEvolution);

    cudaThreadSynchronize();
//DumpDeviceVector(bridgeVariates_device);
//DumpDeviceVector(correlatedVariates_device);

    multi_dim_BS_path_generator_device( initial_log_stocks_device,
        correlatedVariates_device,
        evolved_stocks_device, 
        paths,
        stocks,
        stepsForEvolution);




}

float getPayoffAverage( thrust::device_vector<float>& evolved_stocks_device,
                       thrust::device_vector<float>& averages_device,
                       thrust::device_vector<float>& weights_device,
                       int stocks, 
                       int paths,
                       int steps,
                       thrust::device_vector<float>& auxData_dev,
                       thrust::device_vector<float>& payoffs_device)
{


    basketWeightings_main(evolved_stocks_device, 
        averages_device, 
        weights_device,
        paths,
        stocks,
        steps);
//DumpDeviceVector(averages_device);


    payoffGeneratorCallerRatchetPut_main(payoffs_device,  
        auxData_dev, 
        paths, 
        steps,
        averages_device
        );
  //  DumpDeviceVector(payoffs_device);





    float value = thrust::reduce(payoffs_device.begin(), payoffs_device.end())/paths;


    return value;
}

void setupData( thrust::device_vector<unsigned int>& scrambler_device,
               thrust::device_vector<float>& pseudoRoots_device,
               thrust::device_vector<float>& fixedDrifts_device,
               thrust::device_vector<float>& initial_stocks_device,
               thrust::device_vector<float>& initial_log_stocks_device, 
               thrust::device_vector<int>& alive_device,
               thrust::device_vector<float>& weights_device,
               thrust::device_vector<float>& auxData_dev,
               int stocks,
               int steps)
{

    thrust::fill(scrambler_device.begin(),scrambler_device.end(),0);
    thrust::fill(alive_device.begin(),alive_device.end(),0);

    double rho=0.5;

    double sigma =0.2;
    double S=100.0;
    double r=0.0;
    double d=0.0;
    double dt =0.5;
    double K = S;



    std::vector<double> covarianceMatrix_vec(stocks*stocks);
    std::vector<double> pseudorootMatrix_vec(stocks*stocks);

    MatrixFacade<double> covarianceMatrix_mat(covarianceMatrix_vec,stocks,stocks);
    MatrixFacade<double> pseudorootMatrix_mat(pseudorootMatrix_vec,stocks,stocks);

    for (int i=0; i < stocks; ++i)
    {
        covarianceMatrix_mat(i,i) = sigma*sigma*dt;
        for (int j=0; j < i; ++j)
            covarianceMatrix_mat(i,j) = covarianceMatrix_mat(j,i) = sigma*sigma*rho;
    }

    // now need to compute square root... 

    pseudoSqrtSpectral(
        covarianceMatrix_mat ,
        pseudorootMatrix_mat);



    thrust::host_vector<float> pseudoroots_host(stocks*stocks*steps);
    CubeFacade<float> pseudoroots_host_cube(&pseudoroots_host[0],steps,stocks,stocks);
    for (int i=0; i < steps; ++i)
        for (int j=0; j < stocks; ++j)
            for (int k=0; k < stocks; ++k)
                pseudoroots_host_cube(i,j,k) = static_cast<float>(pseudorootMatrix_mat(j,k));




    pseudoRoots_device = pseudoroots_host;




    thrust::host_vector<float> fixeddrifts_host(stocks*steps);

    MatrixFacade<float> fixedDrifts_mat(&fixeddrifts_host[0],steps,stocks);


    for (int i=0; i < steps; ++i)
        for (int j=0; j < stocks; ++j)
            fixedDrifts_mat(i,j) = static_cast<float>((r-d-0.5*sigma*sigma)*dt);

    fixedDrifts_device = fixeddrifts_host;

    thrust::fill(initial_stocks_device.begin(),initial_stocks_device.end(),S);

    thrust::fill(initial_log_stocks_device.begin(),initial_log_stocks_device.end(),log(S));

    thrust::fill(weights_device.begin(),weights_device.end(),1.0/stocks);

    auxData_dev.resize(1);

    auxData_dev[0]= K;



}




void RatchetPricerExample()
{
    {
        chooseDevice();

        size_t pathsPerBatch = 8*65536;

        int pathOffset=0;

        int stocks=10;
        int factors=10; 
        int stepsForEvolution=64;

        int powerOfTwoForVariates=6;

        int numberBatches = 1;
        std::vector<float> values(numberBatches);

         size_t estimatedFloatsRequired = 6*pathsPerBatch*stepsForEvolution*(factors+stocks);

    size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

    //  bool change=false;

    ConfigCheckForGPU checker;

    float excessMem=2.0f;

    while (excessMem >0.0f)
    {
        checker.checkGlobalMem( estimatedMem, excessMem );

        if (excessMem >0.0f)
        {
            pathsPerBatch/=2;
           estimatedFloatsRequired = 6*pathsPerBatch*stepsForEvolution*(factors+stocks);
           estimatedMem = sizeof(float)*estimatedFloatsRequired;

            std::cout<< "halving batch size to maintain memory space in Ratchet example \n";

        }
    }
       std::cout << " total paths is " << pathsPerBatch*numberBatches << "\n";

        // basic data
        thrust::device_vector<unsigned int> scrambler_device(stepsForEvolution*factors); 
        thrust::device_vector<float> pseudoRoots_device(stocks*factors*stepsForEvolution);
        thrust::device_vector<float> fixedDrifts_device(stocks*factors*stepsForEvolution);
        thrust::device_vector<float> initial_stocks_device(stocks);
        thrust::device_vector<float> initial_log_stocks_device(stocks); 
        thrust::device_vector<int> alive_device(stepsForEvolution);
        thrust::device_vector<float> weights_device(stocks);
        thrust::device_vector<float> auxData_dev(1);


        // buffers for stock evolution
        thrust::device_vector<unsigned int> SobolInts_buffer_device(pathsPerBatch*stepsForEvolution*factors);
        thrust::device_vector<float> quasiRandoms_buffer_device(pathsPerBatch*stepsForEvolution*factors); 
        thrust::device_vector<float> bridgeVariates_device(pathsPerBatch*stepsForEvolution*factors); 
        thrust::device_vector<float> correlatedVariates_device(pathsPerBatch*stepsForEvolution*stocks); 
        thrust::device_vector<float> evolved_stocks_device(pathsPerBatch*stepsForEvolution*stocks);

        setupData(  scrambler_device,
            pseudoRoots_device,
            fixedDrifts_device,
            initial_stocks_device,
            initial_log_stocks_device, 
            alive_device,
            weights_device,
            auxData_dev,
            stocks,
            stepsForEvolution);


        // buffers for pay-off evaluation
        thrust::device_vector<float> evolved_basket_device(pathsPerBatch*stepsForEvolution);
        thrust::device_vector<float> payoffs_device(pathsPerBatch);

        double val=0.0;

        int t1 = clock();

        for (int i=0; i < numberBatches; ++i)
        {


        MultiDBSDoPaths(pathsPerBatch, 
            pathOffset, 
            stocks, 
            factors, 
            stepsForEvolution, 
            powerOfTwoForVariates,
            scrambler_device, 
            pseudoRoots_device,
            fixedDrifts_device, 
            initial_stocks_device, 
            initial_log_stocks_device,  
            alive_device,
            // buffers
            SobolInts_buffer_device, 
            quasiRandoms_buffer_device, 
            bridgeVariates_device, 
            correlatedVariates_device, 
            evolved_stocks_device
            // for output
            );


        values[i] =  getPayoffAverage( evolved_stocks_device,
            evolved_basket_device,
            weights_device,
            stocks, 
            pathsPerBatch,
            stepsForEvolution,
            auxData_dev,
            payoffs_device);

        val += values[i];

        pathOffset+= pathsPerBatch;

  
        }

        int t2 = clock();

        double t = (t2-t1)/static_cast<double>(CLOCKS_PER_SEC);

        std::cout << " time taken is " << t << "\n";

      


        std::cout << "Value is " << val/numberBatches << "\n";

    }
    cudaThreadExit();

}
