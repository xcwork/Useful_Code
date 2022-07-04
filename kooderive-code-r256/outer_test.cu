
//
//
//       outer_test.cu
//
//
// (c) Mark Joshi 2012, 2013, 2014
// This code is released under the GNU public licence version 3
// routine to test the reduced outer product code



#include "outer_test.h"
#include <gold/math/outerProduct_gold.h>
#include <outerProduct_main.h>
#include <ComparisonCheck.h>
#include <cutil_inline.h>
#include <output_device.h>
#include <cutil.h>
#include <vector>
#include <thrust/device_vector.h>
#include <curand.h>
#include <Utilities.h>
#include <reductions_gpu.h>
#include <reduction_thread_fence.h>

double tolerance = 1e-6;

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)

int OuterTest(DeviceChooser& chooser)
{
	int result =4;
	cudaSetDevice(chooser.WhichDevice());
	{

		bool dumpData=false;
		bool dumpData2=true;
		bool singleKernel=false;

		double tolerance = 1E-4;


		int threads = 128;
		int blocks = 64;

		int paths =1 << 15;
		paths*=10;

         size_t estimatedFloatsRequired = 8*paths;
   
         size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

 //  bool change=false;
  
       ConfigCheckForGPU checker;

       float excessMem=2.0f;

       while (excessMem >0.0f)
       {
           checker.checkGlobalMem( estimatedMem, excessMem );

           if (excessMem >0.0f)
           {
               paths/=2;

               estimatedFloatsRequired = 8*paths;
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
     
           }
       }

		int row_size = 10;

		int row_size2= 1;

		int N = paths*row_size;
		int N2 = paths*row_size2;

		std::vector<float> data_vec(N,0.0);
		std::vector<float> data_vec2(N2,0.0);

		MatrixFacade<float> input_data_cpu_mat1(data_vec,row_size,paths);
		MatrixConstFacade<float> input_data_cpu_mat(data_vec,row_size,paths);

		MatrixFacade<float> input_data_cpu_mat2(data_vec2,row_size2,paths);
		MatrixConstFacade<float> input_data_cpu_const_mat2(data_vec2,row_size2,paths);


		for (int i=0; i < row_size; ++i)
			for (int p=0; p < paths; ++p)
				input_data_cpu_mat1(i,p) = //i+p*row_size;
				((i+p+0.1f)/paths)/sqrt(paths+0.0001f);

		for (int i=0; i < row_size2; ++i)
			for (int p=0; p < paths; ++p) 
				input_data_cpu_mat2(i,p) = //100*(i+p*row_size);
				((i+p+0.1f)/paths/100);



		std::vector<float> results_vec(row_size*row_size);

		std::vector<float> results2_vec(row_size*row_size2);

		MatrixFacade<float> results_cpu_mat(results_vec,row_size,row_size);
		MatrixFacade<float> results2_cpu_mat(results2_vec,row_size,row_size2);

		cudaThreadSynchronize();


		Timer h1;
		// gold
		ReducedOuterProductSymmetric( paths,
			row_size,
			input_data_cpu_mat, 
			results_cpu_mat);

		cudaThreadSynchronize();

		double cpuTime =h1.timePassed();

		//      DumpDeviceMatrix(data_vec, row_size, paths);
		//    DumpDeviceMatrix(results_vec, row_size, row_size);


		cudaThreadSynchronize();


		Timer h3;
		//gold
		ReducedOuterProduct( paths,
			row_size,
			row_size2,
			input_data_cpu_mat, 
			input_data_cpu_const_mat2, 
			results2_cpu_mat);

		cudaThreadSynchronize();


		double cpuTime2 =h3.timePassed();



		cudaThreadSynchronize();
		{

			thrust::device_vector<float> input_data_global_dev(data_vec.size());
			thrust::host_vector<float> input_data_host(data_vec.begin(),data_vec.end());

			input_data_global_dev = input_data_host;
			thrust::device_vector<float> answer_global_dev(row_size*row_size);  
			thrust::device_vector<float> answer_global_dev_CU(row_size*row_size);         
			thrust::device_vector<float>  workspace_data_global(blocks);

			thrust::device_vector<float> input_data_global_dev2(data_vec2.size());
			thrust::host_vector<float> input_data_host2(data_vec2.begin(),data_vec2.end());

			input_data_global_dev2 = input_data_host2;

			thrust::device_vector<float> input_data_global_dev3(input_data_global_dev2);

			thrust::device_vector<float> answer_global_dev2(row_size*row_size2);
			thrust::device_vector<float> answer_global_dev2_CU(row_size*row_size2);

			cudaThreadSynchronize();

			Timer h2;

			float alpha = 1.0f;
			float beta = 0.0f;

			reduceOuterProductSymmetricCublas_main( paths,
				row_size,
				alpha,
				beta,
				input_data_global_dev, 
				answer_global_dev_CU);

			cudaThreadSynchronize();

			double gpuTimeCU = h2.timePassed();

			Timer h5;

			reduceOuterProductSymmetric_main( paths,
				row_size,
				threads, 
				blocks, 
				input_data_global_dev, 
				answer_global_dev,
				workspace_data_global);

			cudaThreadSynchronize();
			double gpuTimeHC = h5.timePassed();



	      //  DumpDeviceMatrix(answer_global_dev, row_size, row_size);

			Timer h6;


			reduceOuterProductcublas_main( paths,
				row_size,
				row_size2,
				alpha,beta,
				input_data_global_dev, 
				input_data_global_dev2, 
				answer_global_dev2_CU);

			cudaThreadSynchronize();

			double gpuTime2CU = h6.timePassed();

			Timer h4;

			reduceOuterProduct_main( paths,
				row_size,
				row_size2,
				threads, 
				blocks, 
				singleKernel,
				input_data_global_dev, 
				input_data_global_dev3, 
				answer_global_dev2,
				workspace_data_global);

			cudaThreadSynchronize();
			double gpuTime2HC = h4.timePassed();


			int errors = numberMismatches(results_cpu_mat, answer_global_dev, tolerance, dumpData); // was dumpdata
			int errors2 = numberMismatches(results2_cpu_mat, answer_global_dev2, tolerance*10, dumpData2);

			int errors3 = numberMismatches(results_cpu_mat, answer_global_dev_CU, tolerance, dumpData);
			int errors4 = numberMismatches(results2_cpu_mat, answer_global_dev2_CU, tolerance*10, dumpData);

			if (errors >0)
				std::cout << "symmetric reduced outer product test failed with, " << errors << " errors\n";
			else
			{
				--result;
				std::cout << "symmetric reduced outer product test passed \n";
			}

			if (errors3 >0)
				std::cout << "symmetric reduced outer product CUBLAS test failed with, " << errors3 << " errors\n";
			else
			{
				--result;
				std::cout << "symmetric reduced outer product CUBLAS test passed \n";
			}

			std::cout << "cpu time, " << cpuTime << "," << " gpu time HC, " << gpuTimeHC << " gpu time CU, "<< gpuTimeCU << "\n";
			std::cout << "ratio HC, " << cpuTime/gpuTimeHC << "\n";
			std::cout << "ratio CU, " << cpuTime/gpuTimeCU << "\n";

			if (errors2 >0)
				std::cout << "reduced outer product test failed with, " << errors2 << " errors\n";
			else
			{
				--result;
				std::cout << " reduced outer product test passed \n";
			}

			if (errors4 >0)
				std::cout << "reduced outer product test CUBLAS failed with, " << errors4 << " errors\n";
			else
			{
				--result;
				std::cout << " reduced outer product test CUBLAS passed \n";
			}


			std::cout << "cpu time, " << cpuTime2 << "," << " gpu time HC, " << gpuTime2HC << " gpu time CU, " << gpuTime2CU <<"\n";
			std::cout << "ratio HC, " << cpuTime2/gpuTime2HC << "\n";
			std::cout << "ratio CU, " << cpuTime2/gpuTime2CU << "\n";



		}



	}    
	cudaThreadExit();


	return result;


}

int PointwiseTest(DeviceChooser& chooser)
{
	int result =3;
	cudaSetDevice(chooser.WhichDevice());
	{
		size_t paths =32768;

        		
         size_t estimatedFloatsRequired = 8*paths;
   
         size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

 //  bool change=false;
  
       ConfigCheckForGPU checker;

       float excessMem=2.0f;

       while (excessMem >0.0f)
       {
           checker.checkGlobalMem( estimatedMem, excessMem );

           if (excessMem >0.0f)
           {
               paths/=2;

               estimatedFloatsRequired = 8*paths;
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
     
           }
       }


		thrust::device_vector<float> data1_dev(paths);
		thrust::device_vector<float> data2_dev(paths);
		thrust::device_vector<float> output_dev(paths);

        std::cout << " gen paths using CURAND\n";
		curandGenerator_t gen;

		CURAND_CALL(curandCreateGenerator(&gen, 
			CURAND_RNG_PSEUDO_MTGP32));

		/* Set seed */
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
			1234ULL));

		/* Generate n floats on device */
		CURAND_CALL(curandGenerateUniform(gen, thrust::raw_pointer_cast(&data1_dev[0]), paths));
		CURAND_CALL(curandGenerateUniform(gen, thrust::raw_pointer_cast(&data2_dev[0]), paths));


		int blocks=64;
		int threads=1024;
       std::cout << " do pointwise prod on GPU\n";

		double t1= PointwiseProduct_main(blocks,
			threads,
			paths,
			data1_dev,
			data2_dev,
			output_dev);

		std::vector<float> data1_vec(stlVecFromDevVec(data1_dev));
		std::vector<float> data2_vec(stlVecFromDevVec(data2_dev));
		std::vector<float> output_vec(stlVecFromDevVec(output_dev));

		std::vector<float> output_gold_vec(paths);
           std::cout << " do pointwise prod on CPU\n";

		Timer goldTimer;

		for (size_t i=0; i < paths;++i)
        {
			output_gold_vec[i] = data1_dev[i]*data2_dev[i];

            if (i % 100000 ==0)
                std::cout << i << "\n";
        }
		double t_gold = goldTimer.timePassed();

		double totalErr = 0.0;
		for (size_t i=0; i < paths; ++i)
		{
			totalErr += fabs(output_gold_vec[i] - output_vec[i]);
		}
		double l1Err = totalErr/paths;

		if (l1Err > tolerance)
			std::cout << " pointwise product test failed";
		else
		{
			--result;
			std::cout << " pointwise product test passed.";

		}

		std::cout << " L1 err is " << l1Err << "\n";
		std::cout << " time1 " << t1 << " time gold " << t_gold << "\n";
		std::cout << " ratio " << t_gold/t1 << "\n";

		{
			int reduce8threads=1024; 
			int stopPoint = 1;

			thrust::device_vector<float> wsp_dev(paths);
			float* it = thrust::raw_pointer_cast(&wsp_dev[0]);

			Timer h8;

			const float* ans_loc = PartialReduce_using8s_gpu( thrust::raw_pointer_cast(&output_dev[0]),
				it, // at least size points
				paths,
				reduce8threads);

		   int i = ans_loc - it;

			float answer;
			if (stopPoint >1)
				answer= thrust::reduce(wsp_dev.begin()+i,wsp_dev.begin()+i+stopPoint);
			else
				answer = wsp_dev[i];

			double t8 = h8.timePassed();


			Timer hThrust;

			float answer2 = thrust::reduce(output_dev.begin(),output_dev.end());

			double tthrust = hThrust.timePassed();

			std::cout << " partial 8 reduce test";

			if (fabs((answer-answer2)/answer) > tolerance)
				std::cout <<" failed " << answer << " " << answer2 << "\n";
			else
			{
				std::cout <<  "passed\n";
				--result;
			}    

			std::cout << " thrust time " << tthrust << "\n";
			std::cout << " time eighter " << t8 << "\n";

		}
		{
			int reduceTHthreads=128; 
			int THblocks = 64;

			thrust::device_vector<float> wsp_dev(reduceTHthreads*THblocks);
			float* it = thrust::raw_pointer_cast(&wsp_dev[0]);

			Timer h16;

	//		const float* ans_loc = PartialReduce_using16s_gpu( thrust::raw_pointer_cast(&output_dev[0]),
		//		it, // at least size points
			//	paths,
				//reduce16threads);

			reduceSinglePass(paths, reduceTHthreads, THblocks,  thrust::raw_pointer_cast(&output_dev[0]), it,it);

			

			double t16 = h16.timePassed();
			float answer = wsp_dev[0];
		
			Timer hThrust;

			float answer2 = thrust::reduce(output_dev.begin(),output_dev.end());

			double tthrust = hThrust.timePassed();

			std::cout << " thread fence reduce test ";

			if (fabs((answer-answer2)/answer) > tolerance)
				std::cout <<" failed " << answer << " " << answer2 << "\n";
			else
			{
				std::cout <<  "passed\n";
				--result;
			}    

			std::cout << " thrust time " << tthrust << "\n";
			std::cout << " time thread fence " << t16 << "\n";
			std::cout << " thrust ratio " << tthrust/t16<< "\n";
			std::cout << " gold ratio " << t_gold/t16 << "\n";

		}
	}

	cudaThreadExit();


	return result;


}

