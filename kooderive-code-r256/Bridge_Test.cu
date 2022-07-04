//
//
//                                                   Bridge_Test.cu
//
//
// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3

#include "Bridge_Test.h"
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
#include "cudaMacros.h"

#include "Utilities.h"
#include "Brownian_bridge.h"
#include <gold/Bridge_allocator_gold.h>
// compare GPU against CPU

namespace
{
    double tolerance = 2E-7;

    
}

int BridgeTestRoutine(bool verbose, bool doNormalInSobol, DeviceChooser& chooser)
{
    int retValue =0;
    unsigned int path_offset=0;
    {
        if (verbose)
            std::cout << "\n\nTesting one dimensional Brownian bridge.";




        size_t n_vectors = intPower(2,24);
        int n_poweroftwo =6;
        int n_dimensions = intPower(2,n_poweroftwo);

        size_t N= n_vectors*n_dimensions; 

                   
        cudaSetDevice( chooser.WhichDevice());


              size_t estimatedFloatsRequired = 4*(2*N+2*n_vectors);
        size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

  // bool change=false;
  
       ConfigCheckForGPU checker;

       float excessMem=2.0f;

       while (excessMem >0.0f)
       {
           checker.checkGlobalMem( estimatedMem, excessMem );

           if (excessMem >0.0f)
           {
               n_vectors/=2;
               N= n_vectors*n_dimensions;  
            //   change =true;
               estimatedFloatsRequired = 4*(2*N+2*n_vectors);
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
               if (verbose)
                   std::cout << " checkinh ";

           }
       }
        cudaThreadSynchronize();

        thrust::device_vector<float> dev_output(N);    
        thrust::device_vector<float> device_input(N);


        cutilSafeCall(cudaThreadSynchronize());

        cudaThreadSynchronize();

        if  ( doNormalInSobol)
            int res =  SobDevice( n_vectors, n_dimensions, path_offset, device_input,doNormalInSobol);
        else
            int res =  SobDevice( n_vectors, n_dimensions, path_offset, dev_output,doNormalInSobol);


        cudaThreadSynchronize();

        if (!doNormalInSobol)
        {
            thrust::transform(device_input.begin(), device_input.end(),device_input.begin(),inverseCumulativeShawBrickman<float>());
            cudaThreadSynchronize();
        }


        thrust::host_vector<float> unbridgedNumbersHost(N);   
        unbridgedNumbersHost = device_input;
        cudaThreadSynchronize();

        Timer h1;

        bridgeMainDevice(n_vectors, 
            n_poweroftwo,
            device_input,
            dev_output,verbose);

        cutilSafeCall(cudaThreadSynchronize());
        double Time = h1.timePassed();

        std::cout << "GPU time taken " << Time << "\n";




        cudaThreadSynchronize();

        thrust::host_vector<float> bridgedNumbersHost(N);   
        bridgedNumbersHost = dev_output;





        BrownianBridge<float> BB(n_poweroftwo);
        std::vector<float> variates(n_dimensions);
        std::vector<float> bridgedVariates(n_dimensions);

        double err=0.0;

        for (int i=0; i < n_vectors; ++i)
        {
            for (int d=0; d < n_dimensions; ++d)
                variates[d] = unbridgedNumbersHost[i+d*n_vectors];

            BB.GenerateBridge(variates,bridgedVariates);

            for (int d=0; d < n_dimensions; ++d)
            {
                float bvGold = bridgedVariates[d] ;
                if (d  >0)
                    bvGold -= bridgedVariates[d-1] ;
                float bvGPU =  bridgedNumbersHost[i+d*n_vectors];
                float te = bvGold - bvGPU;

                err += fabs(te);
            }

        }


        double L1err = err/N;


        if (verbose)
        {
            std::cout << "L1  error " << L1err << "\n";
        }

        if (L1err > tolerance)
        {
            std::cout << " Brownian bridge test failed";
            retValue =1;
        }
    }

    cudaThreadExit();


    return retValue;
}


int MultiDBridgeTestOrderingRoutine(bool verbose, bool useTextures, DeviceChooser& chooser)
{
    int result =1;
    {
        if (verbose)
        {
            std::cout << "\n\nTesting multi dimensional Brownian bridge reordering ";
            if (!useTextures)
                std::cout << " not ";
            std::cout << " using textures.\n";
        }   

        bool extraVerbose = false;

        int n_vectors =  intPower(2,15);
        int n_poweroftwo =6;
        int factors = 5;


        int n_steps = intPower(2,n_poweroftwo);

        int tot_dimensions = n_steps*factors;

        size_t N= n_vectors*tot_dimensions; 

        	
            
         size_t estimatedFloatsRequired = 4*N;
   
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
               estimatedFloatsRequired = 4*N;
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
     
           }
       }
           cudaSetDevice( chooser.WhichDevice());
      
        thrust::host_vector<float> test_data(N);
        for (int i=0; i < N; ++i)
            test_data[i] = static_cast<float>(i);


        cudaThreadSynchronize();

        thrust::device_vector<float> dev_output(N);    
        thrust::device_vector<float> device_input(N);
        device_input = test_data;




        cudaThreadSynchronize();


        thrust::host_vector<float> unbridgedNumbersHost(N);   
        unbridgedNumbersHost=device_input ;

        if (extraVerbose)
            for (int i=0; i < n_vectors; ++i)
            {
                for (int d=0; d < tot_dimensions; ++d)
                    std::cout << unbridgedNumbersHost[i+d*n_vectors] << ",";
                std::cout << "\n";
            }

            cudaThreadSynchronize();



            BrownianBridgeMultiDim<float>::ordering allocator( BrownianBridgeMultiDim<float>::triangular);

            // Create a timer to measure performance


           Timer h1;

            MultiDBridgeReordering(n_vectors, 
                n_poweroftwo,
                factors,
                device_input, 
                dev_output,
                allocator,
                useTextures)
                ;


            cudaThreadSynchronize();

			double time = h1.timePassed();
            std::cout << "       time taken for reordering: " 
                << time <<  std::endl;

            thrust::host_vector<float> bridgedNumbersHost(N);   
            bridgedNumbersHost = dev_output;

            if (extraVerbose)
            {
                std::cout << "\n\nbridged numbers\n";
                for (int i=0; i < n_vectors; ++i)
                {
                    for (int f=0; f < factors; ++f)
                    {
                        int vecno = i*factors+f;

                        for (int k=0; k < n_steps; ++k)
                        {
                            std::cout << bridgedNumbersHost[vecno+k*n_vectors*factors] << ",";


                        }

                        std::cout << "\n";
                    }
                }

                std::cout << "\n";
            }



            BrownianBridgeMultiDim<float> BB(n_poweroftwo, factors, allocator);
            std::vector<float> variates(tot_dimensions);
            std::vector<float> bridgedVariates(tot_dimensions);

            double err=0.0;

            for (int i=0; i < n_vectors; ++i)
            {
                for (int d=0; d < tot_dimensions; ++d)
                    variates[d] = unbridgedNumbersHost[i+d*n_vectors];

                BB.reorder(&variates[0],&bridgedVariates[0]);

                for (int f=0; f < factors; ++f)
                {
                    for (int d=0; d < n_steps; ++d)
                    {
                        int r = d+f*n_steps;
                        int vecno = i*factors+f;
                        float bvGold = bridgedVariates[r] ;

                        float bvGPU =  bridgedNumbersHost[vecno+factors*n_vectors*d];

                        if (extraVerbose)
                            std::cout << bvGold<< ",";

                        float te = bvGold - bvGPU;
                        err += fabs(te);
                    }
                    if (extraVerbose)
                        std::cout << "\n";
                }





            }


            double L1err = err/N;


            if (verbose)
            {
                std::cout << "L1  error " << L1err << "\n";
            }

            if (L1err > tolerance)
            {
                std::cout << " MultiD Brownian bridge reordering test failed";
            }
            else
                result =0;
    }

    cudaThreadExit();


    return result;
}




int MultiDBridgeTestRoutine(bool verbose, bool doNormalInSobol, bool useTextures, DeviceChooser& chooser)
{
    int result =1;
    unsigned path_offset=0;
    {
        if (verbose)
            std::cout << "\n\nTesting multi dimensional Brownian bridge.\n";

        bool extraVerbose = false;

        int n_vectors =  intPower(2,23);
        int n_poweroftwo =6;
        int factors = 5;


        int n_steps = intPower(2,n_poweroftwo);

        size_t tot_dimensions = n_steps*factors;

        size_t N= n_vectors*tot_dimensions; 
        
        cudaSetDevice( chooser.WhichDevice());


        size_t estimatedFloatsRequired = 4*(2*N+2*n_vectors);
        size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

  // bool change=false;
  
       ConfigCheckForGPU checker;

       float excessMem=2.0f;

       while (excessMem >0.0f)
       {
           checker.checkGlobalMem( estimatedMem, excessMem );

           if (excessMem >0.0f)
           {
               n_vectors/=2;
               N= n_vectors*tot_dimensions;  
            //   change =true;
               estimatedFloatsRequired = 4*(2*N+2*n_vectors);
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
               if (verbose)
                   std::cout << " checkinh ";

           }
       }


        cudaThreadSynchronize();

        thrust::device_vector<float> dev_output(N);    
        thrust::device_vector<float> device_input(N);

        if  ( doNormalInSobol)
            int res =  SobDevice( n_vectors, tot_dimensions, path_offset, device_input,doNormalInSobol);
        else
            int res =  SobDevice( n_vectors, tot_dimensions, path_offset, dev_output,doNormalInSobol);


        cudaThreadSynchronize();

        if (!doNormalInSobol)
        {
            thrust::transform(device_input.begin(), device_input.end(),device_input.begin(),inverseCumulativeShawBrickman<float>());
            cudaThreadSynchronize();
        }



        cudaThreadSynchronize();


        thrust::host_vector<float> unbridgedNumbersHost(N);   
        unbridgedNumbersHost=device_input ;

        if (extraVerbose)
            for (int i=0; i < n_vectors; ++i)
            {
                for (int d=0; d < tot_dimensions; ++d)
                    std::cout << unbridgedNumbersHost[i+d*n_vectors] << ",";
                std::cout << "\n";
            }

            cudaThreadSynchronize();



            BrownianBridgeMultiDim<float>::ordering allocator( BrownianBridgeMultiDim<float>::triangular);


            MultiDBridge(n_vectors, 
                n_poweroftwo,
                factors,
                device_input, 
                dev_output,
                allocator,
                useTextures)
                ;


            cudaThreadSynchronize();

            thrust::host_vector<float> bridgedNumbersHost(N);   
            bridgedNumbersHost = dev_output;

            if (extraVerbose)
            {
                std::cout << "\n\nbridged numbers\n";
                for (int i=0; i < n_vectors; ++i)
                {
                    for (int f=0; f < factors; ++f)
                    {
                        int vecno = i*factors+f;

                        for (int k=0; k < n_steps; ++k)
                        {
                            std::cout << bridgedNumbersHost[vecno+k*n_vectors*factors] << ",";


                        }

                        std::cout << "\n";
                    }
                }

                std::cout << "\n";
            }



            BrownianBridgeMultiDim<float> BB(n_poweroftwo, factors, allocator);
            std::vector<float> variates(tot_dimensions);
            std::vector<float> bridgedVariates(tot_dimensions);

            int t0=clock();

            for (int i=0; i < n_vectors; ++i)
            {
                for (int d=0; d < tot_dimensions; ++d)
                    variates[d] = unbridgedNumbersHost[i+d*n_vectors];

                BB.reorder(&variates[0],&bridgedVariates[0]);
            }

            int t1=clock();

            std::cout << "time taken for same number of CPU bridges reorderings " << (t1-t0+0.0)/CLOCKS_PER_SEC << "\n";

            double err=0.0;

            for (int i=0; i < n_vectors; ++i)
            {
                for (int d=0; d < tot_dimensions; ++d)
                    variates[d] = unbridgedNumbersHost[i+d*n_vectors];

                BB.GenerateBridge(&variates[0],&bridgedVariates[0]);

                for (int f=0; f < factors; ++f)
                {
                    for (int d=0; d < n_steps; ++d)
                    {
                        int r = d+f*n_steps;
                        int vecno = i*factors+f;
                        float bvGold = bridgedVariates[r] ;

                        float bvGPU =  bridgedNumbersHost[vecno+factors*n_vectors*d];

                        if (extraVerbose)
                            std::cout << bvGold<< ",";

                        float te = bvGold - bvGPU;
                        err += fabs(te);
                    }
                    if (extraVerbose)
                        std::cout << "\n";
                }





            }


            double L1err = err/N;


            if (verbose)
            {
                std::cout << "L1  error " << L1err << "\n";
            }

            if (L1err > tolerance)
            {
                std::cout << " MultiD Brownian bridge  test failed";
            }
            else
                result =0;
    }

    cudaThreadExit();


    return result;
}

int Brownian_bridge_test_routine(bool verbose,  bool justGPU, DeviceChooser& chooser)
{
    int result=1;
    {

        int paths = intPower(2,20);  
        //      1439744;
        int factors = 5;
        int steps = 64;
        int path_offset =0;


        bool extraVerbose = false;


        int dimension = steps*factors;

        std::vector<double> sumsVec(dimension);

        std::vector<double> sumspVec(dimension*dimension,0.0);
        MatrixFacade<double> sumsMat(&sumspVec[0],dimension,dimension);

        std::vector<double> sumsqVec(dimension*dimension,0.0);
        MatrixFacade<double> sumsqMat(&sumsqVec[0],dimension,dimension);


        size_t N= paths*dimension; 

        if (verbose)
            std::cout << " entering Brownian_bridge_test_routine\n";

           cudaSetDevice( chooser.WhichDevice());
        {
                 size_t estimatedFloatsRequired = 4*(2*N+2*paths);
        size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

  // bool change=false;
  
       ConfigCheckForGPU checker;

       float excessMem=2.0f;

       while (excessMem >0.0f)
       {
           checker.checkGlobalMem( estimatedMem, excessMem );

           if (excessMem >0.0f)
           {
               paths/=2;
               N= paths*dimension;  
            //   change =true;
               estimatedFloatsRequired = 4*(2*N+2*paths);
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
               if (verbose)
                   std::cout << " checkinh ";

           }
       }


            cudaThreadSynchronize();

            thrust::device_vector<float> variates_dev(N);    

            thrust::device_vector<float> bridgedVariates_dev(N);  
            std::vector<float> variates_vec(N);

            bool doNormalInSobol = true;

            SobDevice( paths, dimension, path_offset, variates_dev,doNormalInSobol);

            cudaThreadSynchronize();
            {

                thrust::host_vector<float> variates_host(variates_dev);

                std::copy(variates_host.begin(),variates_host.end(), variates_vec.begin());
            }

            std::vector<int> indices(dimension);
            bridge_allocate_diagonal(indices,factors,steps);

            brownian_bridge bb(steps, indices);

			Timer h1;

            bb.transform(variates_dev, 
                bridgedVariates_dev,
                paths, 
                steps, 
                factors);

            cudaThreadSynchronize();

            double time = h1.timePassed();
            std::cout << "GPU time taken " << time << "\n";

            if (justGPU)
                result=0;
            else

            {


                thrust::host_vector<float> bridgedVariates_host(bridgedVariates_dev);
                std::vector<float> bridgedVariates_gpu_vec(bridgedVariates_host.size());
                std::copy(bridgedVariates_host.begin(),bridgedVariates_host.end(), bridgedVariates_gpu_vec.begin());

                std::vector<float> bridgedVariates_cpu_vec(N);

                double t0=clock();

                bb.transform(variates_vec,bridgedVariates_cpu_vec,paths, 
                    steps, 
                    factors);

                double t1=clock();

                double timeC= (t1-t0)/CLOCKS_PER_SEC;
                std::cout << "CPU time taken " << timeC << "\n";
                std::cout << "ratio " << timeC/time << "\n";

                CubeConstFacade<float> bridgedVariates_gpu_cube(&bridgedVariates_gpu_vec[0],steps,factors,paths);
                CubeConstFacade<float> bridgedVariates_cpu_cube(&bridgedVariates_cpu_vec[0],steps,factors,paths);

                double err =0.0;

                for (int p=0; p < paths; ++p)
                {
                    for (int s=0; s < steps; ++s)
                        for (int f=0; f < factors; ++f)
                        {

                            double x = bridgedVariates_cpu_cube(s,f,p);
                            double y = bridgedVariates_gpu_cube(s,f,p);
                            err += fabs( y-x);                   
                        }
                }
                double L1err = ((err/paths)/steps)/factors;

                if ( L1err > tolerance)
                {
                    std::cout << " \nBrownian bridge comparison test failed.\n L1Err = "<< L1err << "\n";        
                }
                else
                {
                    result = 0;
                    if (verbose)
                        std::cout << " \nBrownian bridge comparison test passed. L1Err = "<< L1err << "\n";    

                }

                bool statisticalErrorFound= false;
                std::cout << " \nEntering statistical test.\n";

                for (int s=0; s < steps; ++s)
                {
                    for (int p=0; p < paths; ++p)
                    {

                        for (int d1 =0; d1 < factors; ++d1)
                        {
                            double x= bridgedVariates_gpu_cube(s,d1,p);
                            sumsVec[d1] += x;

                            for (int d2=0; d2< factors; ++d2)
                            {
                                double y= bridgedVariates_gpu_cube(s,d2,p);
                                sumsMat(d1,d2)+= x*y;
                                sumsqMat(d1,d2) += x*x*y*y;
                            }
                        }
                    }

                    std::vector<double> means(factors);

                    double maxErr=0.0;

                    for (int i=0; i < factors; ++i)
                    {
                        means[i] = sumsVec[i]/paths;
                        maxErr=std::max(maxErr,fabs(means[i]));
                    }

                    std::vector<double> covs(dimension*dimension);
                    MatrixFacade<double> covsMat(&covs[0],dimension,dimension);


                    for (int d1 =0; d1 < factors; ++d1)    
                    {
                        if (extraVerbose)
                            std::cout <<"\n" <<  d1 << ","; 

                        for (int d2=0; d2< factors; ++d2)
                        {  
                            sumsMat(d1,d2)/=paths;
                            sumsqMat(d1,d2) /= paths;
                            covsMat(d1,d2) =  sumsMat(d1,d2) - means[d1]*means[d2];

                            if (extraVerbose)
                                std::cout <<  covsMat(d1,d2) << ", ";
                        }
                    }



                    std::vector<double> sterrs(dimension);

                    for (int i=0; i < factors; ++i)
                    {
                        sterrs[i] =  sqrt(covsMat(i,i)/paths);

                        if (fabs(means[i]) > 5*sterrs[i])
                        {
                            std::cout << i << " mean " << means[i] << " st err " << sterrs[i] << "\n";
                            statisticalErrorFound = true;
                        }

                        double covStErr = sqrt(1.0/paths);

                        for (int j=0; j <i; ++j)
                            if (covsMat(i,j) > 4*covStErr)
                            {
                                std::cout << i << " " << j << " covariance " << covsMat(i,j) << " st err " << covStErr << "\n";
                                statisticalErrorFound = true;
                            }

                            if (fabs(covsMat(i,i)-1.0) > 6*covStErr)
                            {
                                std::cout << i << " " << i << " covariance " << covsMat(i,i) << " st err " << sqrt(2.0)*covStErr << "\n";
                                statisticalErrorFound = true;
                            }
                    }


                }


                if (statisticalErrorFound)
                {
                    ++result;
                }
                else 
                    std::cout << "Statistical test passed.\n";
            }

        }

    }
    cudaThreadExit();

    return result;
}



