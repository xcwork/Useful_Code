//
//
//       LMM_Evolver_Test.cu
//
//
// (c) Mark Joshi 2010, 2013,2014
// This code is released under the GNU public licence version 3


#include "LMM_evolver_test.h"
#include "LMM_evolver_main.h"
#include <gold/LMM_evolver_gold.h>
#include <gold/correlate_drift_gold.h>
#include "multid_path_gen_BS_main.h"

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
#include "correlate_drift_main.h"
#include "cudaMacros.h"
#include <gold/MatrixFacade.h> 

#include "Utilities.h"
#include "cashFlowGeneration_product_main.h"
#include <gold/cashFlowGeneration_product_gold.h>
#include <gold/cashFlowDiscounting_gold.h>
#include "cashFlowDiscounting_main.h"
#include <gold/volstructs_gold.h>
// compare GPU against CPU

namespace
{
	double tolerance = 2E-6;

}


int LMMLogEulerTestRoutine(bool verbose,  bool useTextures, DeviceChooser& chooser)
{
	int path_offset=0;
	int result =13; // 13 tests have yet to be passed 
	{
		if (verbose)
		{
			std::cout << "\n\nTesting LMM Euler evolver routine ";
			if (!useTextures)
				std::cout << " not ";
			std::cout << " using textures where there's a choice.\n";
		}
		bool extraVerbose = false;

		cudaSetDevice(chooser.WhichDevice());

		thrust::device_vector<int> alive_device;   

		int n_vectors =   intPower(2,21);
#ifdef _DEBUG
		n_vectors = 64;
#endif
		int n_poweroftwo =3;
		int factors = 5;
		int n_steps = intPower(2,n_poweroftwo);
		int tot_dimensions = n_steps*factors;
		int N= n_vectors*tot_dimensions; 
		int number_rates = n_steps;

		int steps_to_test  = n_steps; // must be less than or equal to n_steps


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



		std::vector<int> aliveIndex(n_steps);
		for (int i=0; i < n_steps; ++i)
			aliveIndex[i] = i;



		thrust::device_vector<float> evolved_rates_device(outN);
		std::vector<float> evolved_rates_cpu_vec(n_vectors*number_rates*n_steps);

		int numberSteps = n_steps;
		int pathsPerBatch = n_vectors;
		double beta_d = 0.2;
		double L_d = 0.0;


		// dummy calibration data 

		std::vector<float> logRates_vec(number_rates);
		std::vector<float> rates_vec(number_rates);
		std::vector<float> taus_vec(number_rates);
		std::vector<float> displacements_vec(number_rates);
		std::vector<float> rateTimes_vec(number_rates+1);
		std::vector<float> evolutionTimes_vec(number_rates);

		std::vector<double> rateTimes_d_vec(number_rates+1);
		std::vector<double> evolutionTimes_d_vec(number_rates);

		rateTimes_vec[0] =0.5;
		rateTimes_d_vec[0] =0.5;

		for (int i=0; i < number_rates; ++i)
		{
			rates_vec[i] = 0.05f;
			displacements_vec[i] =0.02f;
			logRates_vec[i] = logf(rates_vec[i]+displacements_vec[i] );
			taus_vec[i] = 0.5f;
			rateTimes_d_vec[i+1] = 0.5+0.5*(i+1);
			rateTimes_vec[i+1]= 0.5f+0.5f*(i+1);
			evolutionTimes_d_vec[i] = 0.5+i*0.5; 
			evolutionTimes_vec[i] = rateTimes_vec[i];

		}   

		float maxDisplacement = *std::max_element(displacements_vec.begin(),displacements_vec.end());

		float vol = 0.11f;

		std::vector<float> vols(number_rates,vol);
		std::vector<double> vols_d(number_rates,vol);

		thrust::host_vector<float> taus_host(taus_vec.begin(),taus_vec.end());
		thrust::device_vector<float> taus_device(taus_host);


		thrust::host_vector<float> displacements_host(displacements_vec.begin(),displacements_vec.end());
		thrust::device_vector<float> displacements_device(displacements_host);
		std::vector<float> evolved_rates_gpu_vec(evolved_rates_device.size());

		CubeConstFacade<float> forwards_cube(&evolved_rates_gpu_vec[0],numberSteps,number_rates,pathsPerBatch);

		thrust::device_vector<float> discounts_device(n_vectors*steps_to_test*(number_rates+1));
		thrust::host_vector<float> discounts_host(discounts_device.size());
		std::vector<float> discounts_vec(discounts_host.size());

		CubeConstFacade<float> discountRatios_cube(&discounts_host[0],numberSteps,number_rates+1,pathsPerBatch);

		// cash-flow aggregation data

		//		thrust::device_vector<float> aggregatedFlows_device(totalPaths*exerciseIndices_vec.size());




		//		std::vector<float> aggregatedFlows_vec(totalPaths*exerciseIndices_vec.size());
		//		MatrixFacade<float> aggregatedFlows_cpu_matrix(&aggregatedFlows_vec[0],exerciseIndices_vec.size(),totalPaths);


		// create new scope so that everything inside dies at end of it
		{

			thrust::host_vector<int> alive_host(aliveIndex.begin(),aliveIndex.end());
			alive_device=alive_host;


			/*	Cube_gold<float> pseudos(FlatVolPseudoRootsFloat( rateTimes_vec,
			evolutionTimes_vec,
			beta,
			L,
			factors,
			vols));*/




			Cube_gold<double> pseudosDouble(FlatVolPseudoRootsOfCovariances(rateTimes_d_vec,
				evolutionTimes_d_vec,
				vols_d,
				factors,
				L_d,
				beta_d
				));

			//		debugDumpCube<double>(pseudosDouble,"pseudos");

			Cube_gold<float> pseudos(CubeTypeConvert<float,double>(pseudosDouble));

			thrust::device_vector<float> A_device(deviceVecFromCube(pseudos));



			/*


			std::vector<float> logRates_vec(number_rates);
			std::vector<float> rates_vec(number_rates);
			std::vector<float> taus_vec(number_rates);
			std::vector<float> displacements_vec(number_rates);
			std::vector<float> rateTimes_vec(number_rates+1);
			std::vector<float> evolutionTimes_vec(number_rates);

			rateTimes_vec[0] =0.5;

			for (int i=0; i < number_rates; ++i)
			{
			rates_vec[i] = 0.05f+0.01f*i;
			displacements_vec[i] =0.02f+0.001f*i;
			logRates_vec[i] = logf(rates_vec[i]+displacements_vec[i] );
			taus_vec[i] = 0.5f;
			rateTimes_vec[i+1] =rateTimes_vec[i]+taus_vec[i];
			evolutionTimes_vec[i] = rateTimes_vec[i];

			}   

			float  beta =0.2f;
			float  L =0.0f;
			float vol = 0.11f;
			std::vector<float> vols(number_rates,vol);


			thrust::host_vector<float> taus_host(taus_vec.begin(),taus_vec.end());
			thrust::device_vector<float> taus_device(taus_host);
			std::vector<float> evolved_rates_cpu_vec(n_vectors*number_rates*n_steps);

			thrust::host_vector<float> displacements_host(displacements_vec.begin(),displacements_vec.end());
			thrust::device_vector<float> displacements_device(displacements_host);
			std::vector<float> evolved_rates_gpu_vec;

			// create new scope so that everything inside dies at end of it
			{	
			Cube_gold<float> pseudos(FlatVolPseudoRootsFloat( rateTimes_vec,
			evolutionTimes_vec,
			beta,
			L,
			factors,
			vols));

			thrust::device_vector<float> A_device(deviceVecFromCube(pseudos));


			//std::vector<float> A(number_rates*factors*n_steps);

			/*		{
			int i=0;
			for (int s=0; s < n_steps;++s)
			for (int r=0; r < number_rates; ++r)
			for (int f=0; f< factors; ++f, ++i)
			if (r >= aliveIndex[s])
			A[i] = static_cast<float>((A.size()+i+1.0f)/A.size())/(5.0f*factors);
			else
			A[i] =0.0f;
			}

			CubeConstFacade<float> pseudos(&A[0],n_steps,number_rates,factors);


			thrust::host_vector<float> A_host(A.begin(),A.end());
			thrust::device_vector<float> A_device(A_host.size());
			A_device  = A_host;
			*/
			//		thrust::host_vector<int> alive_host(aliveIndex.begin(),aliveIndex.end());
			//	alive_device=alive_host;





			std::vector<float> drifts(n_steps*number_rates);
			MatrixFacade<float> drifts_mat(&drifts[0],n_steps,number_rates);
			for (int i=0; i < static_cast<int>(drifts.size()); ++i)
			{
				for (int s=0; s < n_steps; ++s)
					for (int r=0; r < number_rates; ++r)
					{  
						float x = 0.0f;
						for (int f=0; f< factors; ++f)
							x += pseudos(s,r,f)*pseudos(s,r,f);

						drifts_mat(s,r) = -0.5f*x;

					}
			}


			thrust::host_vector<float> drifts_host(drifts.begin(),drifts.end());
			thrust::device_vector<float> drifts_device(drifts_host);



			thrust::host_vector<float> logRates_host(logRates_vec.begin(),logRates_vec.end());
			thrust::device_vector<float> logRates_device(logRates_host);

			thrust::host_vector<float> rates_host(rates_vec.begin(),rates_vec.end());
			thrust::device_vector<float> rates_device(rates_host);


			thrust::device_vector<float> dev_output(N);    
			thrust::device_vector<float> device_input(N);

			thrust::device_vector<float> e_buffer_device(factors*n_vectors);

			thrust::device_vector<float> initial_drifts_device(number_rates);

			{

				spotDriftComputer<float> driftComp(taus_vec,  factors,displacements_vec);
				std::vector<float> initial_drifts_vec(number_rates);

				driftComp.getDrifts(pseudos[0], 
					rates_vec,
					initial_drifts_vec);

				thrust::host_vector<float> initialDrifts_host(initial_drifts_vec.begin(),initial_drifts_vec.end());
				initial_drifts_device = initialDrifts_host;

			}





			cudaThreadSynchronize();

			Timer h1;

			bool doNormalInSobol=true;
			SobDevice( n_vectors, tot_dimensions,path_offset, device_input,doNormalInSobol);
			cudaThreadSynchronize();



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




			thrust::device_vector<float> dev_correlated_rates(outN);

			thrust::device_vector<float> evolved_log_rates_device(outN);


			if (extraVerbose)
			{
				std::cout << "logRates_device " << "\n";

				for (int i=0; i < number_rates; ++i)
					std::cout << logRates_device[i] << ",";

				std::cout << "\n ";
				std::cout << "rates_device ";

				for (int i=0; i < number_rates; ++i)
					std::cout << rates_device[i] << ",";

				std::cout << "\n taus device  ";
				for (int i=0; i < number_rates; ++i)
					std::cout << taus_device[i] << ",";

				std::cout << "\n A device \n";
				for (int s=0; s < n_steps; ++s)
				{
					std::cout << s << "\n";
					for (int r=0; r < number_rates; ++r)
					{
						for (int f=0; f < factors; ++f)
							std::cout << A_device[s*number_rates*factors+r*factors+f]<<",";

						std::cout << "\n";
					}

				}
				std::cout << "\n Initial Drifts device  ";

				for (int i=0; i < number_rates; ++i)
					std::cout << initial_drifts_device[i] << ",";

				std::cout << "\n";
			}





			correlated_drift_paths_device( dev_output,
				dev_correlated_rates, // correlated rate increments 
				A_device, // correlator 
				alive_device,
				drifts_device,
				factors*number_rates,
				factors, 
				number_rates,
				n_vectors,
				n_steps);

			LMM_evolver_euler_main(  rates_device, 
				logRates_device, 
				taus_device,
				dev_correlated_rates,
				A_device,
				initial_drifts_device,
				displacements_device,
				aliveIndex,
				n_vectors,
				factors,
				steps_to_test, 
				number_rates, 
				e_buffer_device,
				evolved_rates_device, // for output
				evolved_log_rates_device  // for output 
				);


			double time =h1.timePassed();
			std::cout << " time taken for all steps of path generation."    << time << std::endl;

			thrust::host_vector<float> evolved_rates_host(evolved_rates_device);
			thrust::host_vector<float> evolved_log_rates_host(evolved_log_rates_device);


			evolved_rates_gpu_vec.resize(evolved_rates_host.size());
			std::copy(evolved_rates_host.begin(),evolved_rates_host.end(),evolved_rates_gpu_vec.begin());
			std::vector<float> evolved_log_rates_gpu_vec(evolved_log_rates_host.begin(), evolved_log_rates_host.end());

			std::vector<float> evolved_log_rates_cpu_vec(n_vectors*number_rates*n_steps);



			thrust::host_vector<float> rateIncrements_host(dev_correlated_rates);
			std::vector<float> rateIncrements_vec(rateIncrements_host.begin(),rateIncrements_host.end());



			int t0=clock();

			LMM_evolver_Euler_gold(rates_vec,
				taus_vec,
				rateIncrements_vec, //  AZ  + mu_fixed
				pseudos.getDataVector(),
				displacements_vec,
				n_vectors,
				factors,
				steps_to_test, 
				number_rates, 
				evolved_rates_cpu_vec,
				evolved_log_rates_cpu_vec);

			int t1=clock();

			float time2 = (t1-t0+0.0f)/CLOCKS_PER_SEC;

			std::cout << " time taken for CPU evolution " << time2 << "\n";

			std::cout << " speed up ratio " << time2/time << "\n";



			double err1=0.0;
			double err2 = 0.0;

			int count =0;

			for (int p = 0; p < n_vectors; ++p)
				for (int s=0; s < steps_to_test; ++s)
				{
					if (extraVerbose)
						std::cout << p << "," << s ;

					for (int r=aliveIndex[s]; r < number_rates; ++r)     
					{
						int i = p +    n_vectors*(r+ s*number_rates);
						if (extraVerbose)
							std::cout <<  "," << evolved_rates_gpu_vec[i] ;


						double erri =2* (evolved_rates_gpu_vec[i] - evolved_rates_cpu_vec[i])/(evolved_rates_gpu_vec[i] + evolved_rates_cpu_vec[i]);
						++count;
						err1 += fabs(erri);
						err2 += fabs(erri*erri);

						//						if (erri > 0.01)
						//							std::cout << p << "," << s << "," << r <<"," <<evolved_rates_cpu_vec[i] << "," << evolved_rates_gpu_vec[i]<<"\n";

					}

					if (extraVerbose)
					{

						std::cout << ",,";

						for (int r=0; r < number_rates; ++r)     
						{
							int i = p +    n_vectors*(r+ s*number_rates);
							std::cout <<  "," << evolved_rates_cpu_vec[i] ;
						}

						std::cout << ",,";

						for (int r=0; r < number_rates; ++r)     
						{
							int i = p +    n_vectors*(r+ s*number_rates);
							std::cout <<  "," << evolved_log_rates_gpu_vec[i] ;
						}

						std::cout << ",,";

						for (int r=0; r < number_rates; ++r)     
						{
							int i = p +    n_vectors*(r+ s*number_rates);
							std::cout <<  "," << evolved_log_rates_cpu_vec[i] ;
						}



						std::cout << "\n";
					}
				}




				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}

				if (L1err > tolerance)
				{
					std::cout << " LMM Evolver test failed";
				}
				else
					--result;
		}


		//		thrust::device_vector<float> discounts_device(n_vectors*steps_to_test*(number_rates+1));
		std::vector<float> discounts_cpu_vec(discounts_device.size());

		for (int loop=0; loop < 2; ++loop)
		{


			cudaThreadSynchronize();

			Timer h1;
			bool allStepsAtOnce = loop > 0;


			discount_ratios_computation_main( evolved_rates_device, 
				taus_device, 
				aliveIndex, 
				alive_device,
				n_vectors,
				steps_to_test, 
				number_rates, 
				discounts_device,
				allStepsAtOnce  // for output 
				);


			double time = h1.timePassed();
			std::cout << " time taken for discount ratio computation."    << time << std::endl;

			thrust::host_vector<float> discounts_host(discounts_device);
			std::vector<float> discounts_gpu_vec(discounts_host.begin(), discounts_host.end());

			int t0=clock();

			discount_ratios_computation_gold( evolved_rates_cpu_vec, 
				taus_vec, 
				aliveIndex, 
				n_vectors,
				steps_to_test, 
				number_rates, 
				discounts_cpu_vec  // for output 
				);                                                                                       
			int t1=clock();

			float timeCPU = (t1-t0+0.0f)/CLOCKS_PER_SEC;

			std::cout << " time taken for discount ratio computation on CPU: "    << timeCPU << std::endl;
			std::cout << " speed up " << timeCPU/time << "\n";

			int count =0;

			double err1=0.0f;
			double err2=0.0f;
			for (int p = 0; p < n_vectors; ++p)
				for (int s=0; s < steps_to_test; ++s)
				{
					if (extraVerbose)
						std::cout << p << "," << s ;

					for (int r=aliveIndex[s]; r <= number_rates; ++r)     
					{
						int i = p +    n_vectors*(r+ s*(number_rates+1));
						if (extraVerbose)
							std::cout <<  "," << discounts_gpu_vec[i] ;


						double erri =2* (discounts_gpu_vec[i] - discounts_cpu_vec[i])/(discounts_gpu_vec[i] + discounts_cpu_vec[i]);
						++count;
						err1 += fabs(erri);
						err2 += fabs(erri*erri);
					}

					if (extraVerbose)
					{

						std::cout << ",,";

						for (int r=aliveIndex[s]; r <= number_rates; ++r)     
						{
							int i = p +    n_vectors*(r+ s*(number_rates+1));
							std::cout <<  "," << discounts_cpu_vec[i] ;
						}



						std::cout << "\n";
					}
				}




				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}

				if (L1err > tolerance)
				{
					std::cout << " discount ratios test " << allStepsAtOnce << " failed\n";
				}
				else
				{

					std::cout << " discount ratios test " << allStepsAtOnce << " passed\n";
					--result;
				}
		}
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////

		thrust::device_vector<float> numeraire_values_device(n_vectors*number_rates,-1.0);
		std::vector<float> numeraire_values_vec(numeraire_values_device.size());


		{ // now test spot measure numeraires computations 

			cudaThreadSynchronize();

			Timer h1;

			spot_measure_numeraires_computation_gpu_main(   discounts_device,
				numeraire_values_device, //output
				n_vectors,
				number_rates
				);



			double time = h1.timePassed();
			std::cout << " \n\ntime taken for spot measure computation."    << time << std::endl;

			thrust::host_vector<float> numeraire_values_host(numeraire_values_device);
			std::copy(numeraire_values_host.begin(), numeraire_values_host.end(),numeraire_values_vec.begin());
			std::vector<float> numeraire_values_cpu_vec(numeraire_values_vec.size());

			Timer h2;

			spot_measure_numeraires_computation_gold( discounts_cpu_vec, 
				n_vectors,
				number_rates, 
				numeraire_values_cpu_vec  // for output 
				);                                                                                       

			double time2 = h2.timePassed();


			float timeCPU =static_cast<float>(time2);

			std::cout << " time taken for numeraire computation on CPU: "    << timeCPU << std::endl;
			std::cout << " speed up " << timeCPU/time << "\n\n";

			int count =0;

			double err1=0.0f;
			double err2=0.0f;

			for (int p = 0; p < n_vectors; ++p)
				for (int s=0; s < steps_to_test; ++s)
				{           
					int i= p + s*n_vectors;
					double erri =2* (numeraire_values_vec[i] - numeraire_values_cpu_vec[i])/(numeraire_values_vec[i] + numeraire_values_cpu_vec[i]);
					++count;
					err1 += fabs(erri);
					err2 += fabs(erri*erri);
				}


				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}

				if (L1err > tolerance)
				{
					std::cout << " numeraires test failed";
				}
				else
					--result;
		}
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		// now test annuities code

		thrust::device_vector<float> annuities_device(n_vectors*steps_to_test*number_rates);
		std::vector<float> annuities_vec(n_vectors*steps_to_test*number_rates);

		{


			cudaThreadSynchronize();


			Timer h1;


			coterminal_annuity_ratios_computation_gpu(  discounts_device, 
				taus_device, 
				aliveIndex, 
				n_vectors,
				steps_to_test, 
				number_rates, 
				annuities_device  // for output 
				);

			double time = h1.timePassed();
			std::cout << " time taken for annuity computation."    << time << std::endl;


			thrust::host_vector<float> annuities_host(annuities_device);
			std::vector<float> annuities_gpu_vec(annuities_host.begin(), annuities_host.end());


			int t0=clock();


			coterminal_annuity_ratios_computation_gold(  discounts_cpu_vec, 
				taus_vec, 
				aliveIndex, 
				n_vectors,
				steps_to_test, 
				number_rates, 
				annuities_vec  // for output 
				);

			int t1=clock();

			float timeCPU = (t1-t0+0.0f)/CLOCKS_PER_SEC;

			std::cout << " time taken for annuity computation on CPU: "    << timeCPU << std::endl;
			std::cout << " speed up " << timeCPU/time << "\n";

			int count =0;

			double err1=0.0;
			double err2=0.0;
			for (int p = 0; p < n_vectors; ++p)
				for (int s=0; s < steps_to_test; ++s)
				{
					if (extraVerbose)
						std::cout << p << "," << s ;

					for (int r=aliveIndex[s]; r < number_rates; ++r)     
					{
						int i = p +    n_vectors*(r+ s*number_rates);
						if (extraVerbose)
							std::cout <<  "," << annuities_gpu_vec[i] ;


						double erri =2* (annuities_gpu_vec[i] - annuities_vec[i])/(annuities_gpu_vec[i] + annuities_vec[i]);
						++count;
						err1 += fabs(erri);
						err2 += fabs(erri*erri);
					}

					if (extraVerbose)
					{

						std::cout << ",,";

						for (int r=aliveIndex[s]; r < number_rates; ++r)     
						{
							int i = p +    n_vectors*(r+ s*number_rates);
							std::cout <<  "," << annuities_vec[i] ;
						}



						std::cout << "\n";
					}
				}




				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}

				if (L1err > tolerance)
				{
					std::cout << " annuity test failed";
				}
				else
					--result;
		}

		debugDumpCube(annuities_vec,"annuities_vec", steps_to_test,number_rates,n_vectors);

		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////

		// now test coterminal swap-rate code

		thrust::device_vector<float> cot_swaps_device(n_vectors*steps_to_test*number_rates);
		std::vector<float> cot_swaps_vec(n_vectors*steps_to_test*number_rates);

		{


			cudaThreadSynchronize();


			Timer h1;

			coterminal_swap_rates_computation_main_gpu(  discounts_device, 
				annuities_device, 
				aliveIndex, 
				n_vectors,
				steps_to_test, 
				number_rates, 
				cot_swaps_device  // for output 
				);

			double time = h1.timePassed();
			std::cout << " time taken for cot swap-rate computation."    << time << std::endl;


			thrust::host_vector<float> cot_swaps_host(cot_swaps_device);
			std::vector<float> cot_swaps_gpu_vec(cot_swaps_host.begin(), cot_swaps_host.end());


			int t0=clock();


			coterminal_swap_rates_computation_gold(  discounts_cpu_vec, 
				annuities_vec,
				aliveIndex, 
				n_vectors,
				steps_to_test, 
				number_rates, 
				cot_swaps_vec  // for output 
				);

			int t1=clock();

			float timeCPU = (t1-t0+0.0f)/CLOCKS_PER_SEC;

			std::cout << " time taken for cot swap-rate computation on CPU: "    << timeCPU << std::endl;
			std::cout << " speed up " << timeCPU/time << "\n";

			debugDumpCube(cot_swaps_vec,"cot_swaps_vec", steps_to_test,number_rates,n_vectors);
			int count =0;

			double err1=0.0;
			double err2=0.0;
			for (int p = 0; p < n_vectors; ++p)
				for (int s=0; s < steps_to_test; ++s)
				{
					if (extraVerbose)
						std::cout << p << "," << s ;

					for (int r=aliveIndex[s]; r < number_rates; ++r)     
					{
						int i = p +    n_vectors*(r+ s*number_rates);
						if (extraVerbose)
							std::cout <<  "," << cot_swaps_gpu_vec[i] ;


						double erri =2* (cot_swaps_gpu_vec[i] - cot_swaps_vec[i])/(cot_swaps_gpu_vec[i] + cot_swaps_vec[i]);
						++count;
						err1 += fabs(erri);
						err2 += fabs(erri*erri);
					}

					if (extraVerbose)
					{

						std::cout << ",,";

						for (int r=aliveIndex[s]; r < number_rates; ++r)     
						{
							int i = p +    n_vectors*(r+ s*number_rates);
							std::cout <<  "," << cot_swaps_vec[i] ;
						}



						std::cout << "\n";
					}
				}




				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}

				if (L1err > tolerance)
				{
					std::cout << " swap-rate test failed";
				}
				else
					--result;
		}

		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		// now test swap-rate code

		int swapRatesPerStep = 1;
		thrust::device_vector<float> swap_rate_device(n_vectors*steps_to_test*swapRatesPerStep);
		std::vector<float> swap_rate_gpu_vec(n_vectors*steps_to_test*swapRatesPerStep);

		{

			std::vector<int> startIndex(steps_to_test);
			std::vector<int> endIndex(steps_to_test);

			int tenor = 10;

			for (int j=0; j < steps_to_test; ++j)
			{
				startIndex[j] = aliveIndex[j];
				endIndex[j] = std::min<int>(number_rates, startIndex[j]+tenor);
			}


			cudaThreadSynchronize();


			Timer h1;



			for (int s=0; s< steps_to_test; ++s)
			{
				swap_rate_computation_gpu(  discounts_device, 
					taus_device, 
					startIndex[s],
					endIndex[s], 
					n_vectors,
					s,
					s,
					number_rates,
					swap_rate_device
					);
			}

			double time = h1.timePassed();
			std::cout << " time taken for swap-rate computation."    << time << std::endl;


			thrust::host_vector<float> swap_rate_host(swap_rate_device);
			std::vector<float> swap_rate_gpu_vec2(swap_rate_host.begin(), swap_rate_host.end());      
			swap_rate_gpu_vec = swap_rate_gpu_vec2;
			std::vector<float> swap_rate_cpu_vec(swap_rate_gpu_vec.size());

			int t0=clock();


			for (int s=0; s < steps_to_test; ++s)           
				swap_rate_computation_gold( discounts_cpu_vec, 
				taus_vec, 
				startIndex[s],
				endIndex[s], 
				n_vectors,
				s,
				s ,
				steps_to_test,
				number_rates, 
				swap_rate_cpu_vec  
				);                                                                                            


			int t1=clock();

			float timeCPU = (t1-t0+0.0f)/CLOCKS_PER_SEC;

			std::cout << " time taken for swap rate computation on CPU: "    << timeCPU << std::endl;
			std::cout << " speed up " << timeCPU/time << "\n";

			int count =0;

			double err1=0.0;
			double err2=0.0;
			for (int p = 0; p < n_vectors; ++p)
				for (int s=0; s < steps_to_test; ++s)
				{
					if (extraVerbose)
						std::cout << p << "," << s ;


					int i = p +    n_vectors*s;
					if (extraVerbose)
						std::cout <<  "," << swap_rate_gpu_vec[i] ;


					double erri =2* (swap_rate_gpu_vec[i] - swap_rate_cpu_vec[i])/(swap_rate_gpu_vec[i] + swap_rate_cpu_vec[i]);
					++count;
					err1 += fabs(erri);
					err2 += fabs(erri*erri);


					if (extraVerbose)
					{

						std::cout << ",,";


						int i = p +    n_vectors*s;
						std::cout <<  "," << swap_rate_cpu_vec[i] ;




						std::cout << "\n";
					}

				}




				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}

				if (L1err > tolerance)
				{
					std::cout << " swap test failed";
				}
				else
					--result;
		}

		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////

		thrust::device_vector<float> extracted_forwards_device(steps_to_test*n_vectors);
		std::vector<float> extracted_forwards_cpu_vec(extracted_forwards_device.size());


		// now test forward rate extraction code
		{


			std::vector<int> extractors(steps_to_test);
			for (int i=0; i < static_cast<int>(extractors.size()); ++i)
				extractors[i] = i;


			cudaThreadSynchronize();

			Timer h1;


			forward_rate_extraction_gpu_main(  evolved_rates_device, 
				extractors,                          
				n_vectors,
				steps_to_test,
				number_rates, 
				extracted_forwards_device                
				);


			double time = h1.timePassed();
			std::cout << " time taken for forward_rate_extraction_gpu_main computation."    << time << std::endl;


			thrust::host_vector<float> extracted_forwards_host(extracted_forwards_device);
			std::vector<float> extracted_forwards_vec(extracted_forwards_host.begin(), extracted_forwards_host.end());

			int t0=clock();

			forward_rate_extraction_gold(  evolved_rates_gpu_vec, 
				extractors,                          
				n_vectors,
				steps_to_test,
				number_rates, 
				extracted_forwards_cpu_vec                
				);




			int t1=clock();

			float timeCPU = (t1-t0+0.0f)/CLOCKS_PER_SEC;

			std::cout << " time taken for forward_rate_extraction_gold on CPU: "    << timeCPU << std::endl;
			std::cout << " speed up " << timeCPU/time << "\n";



			int count =0;

			double err1=0.0;
			double err2=0.0;
			for (int p = 0; p < n_vectors; ++p)
				for (int s=0; s < steps_to_test; ++s)
				{
					int i = p +     s*n_vectors;

					if (extraVerbose)
						std::cout << s << "'," << p << "," << extracted_forwards_vec[i] << "," <<extracted_forwards_cpu_vec[i] << "\n";

					double erri =2* (extracted_forwards_vec[i] - extracted_forwards_cpu_vec[i])/(extracted_forwards_vec[i] + extracted_forwards_cpu_vec[i]);
					++count;
					err1 += fabs(erri);
					err2 += fabs(erri*erri);



				}




				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}



				if (L1err > tolerance)
				{
					std::cout << " forward rate extraction test failed";
				}
				else
					--result;
		}
		/////////////////////////////////////////////////////////////////
		// now test forward rate steps extraction code
		{


			std::vector<int> extractors(steps_to_test);
			std::vector<int> steps(steps_to_test);
			for (int i=0; i < static_cast<int>(extractors.size()); ++i)
			{
				extractors[i] = i;
				steps[i] = std::max(i-1,0);
			}

			cudaThreadSynchronize();

			Timer h1;


			forward_rate_extraction_steps_gpu_main(  evolved_rates_device, 
				extractors,            
				steps,
				n_vectors,
				number_rates, 
				extracted_forwards_device                
				);


			double time = h1.timePassed();
			std::cout << " time taken for forward_rate_extraction_steps_gpu_main computation."    << time << std::endl;


			thrust::host_vector<float> extracted_forwards_host(extracted_forwards_device);
			std::vector<float> extracted_forwards_vec(extracted_forwards_host.begin(), extracted_forwards_host.end());

			int t0=clock();

			forward_rate_extraction_selecting_steps_gold(  evolved_rates_gpu_vec, 
				extractors,    
				steps,
				n_vectors,
				number_rates, 
                steps_to_test,
				extracted_forwards_cpu_vec                
				);




			int t1=clock();

			float timeCPU = (t1-t0+0.0f)/CLOCKS_PER_SEC;

			std::cout << " time taken for forward_rate_extraction_gold on CPU: "    << timeCPU << std::endl;
			std::cout << " speed up " << timeCPU/time << "\n";



			int count =0;

			double err1=0.0;
			double err2=0.0;
			for (int p = 0; p < n_vectors; ++p)
				for (int s=0; s < steps_to_test; ++s)
				{
					int i = p +     s*n_vectors;

					if (extraVerbose)
						std::cout << s << "'," << p << "," << extracted_forwards_vec[i] << "," <<extracted_forwards_cpu_vec[i] << "\n";

					double erri =2* (extracted_forwards_vec[i] - extracted_forwards_cpu_vec[i])/(extracted_forwards_vec[i] + extracted_forwards_cpu_vec[i]+2*tolerance);
					++count;
					err1 += fabs(erri);
					err2 += fabs(erri*erri);



				}




				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}



				if (L1err > tolerance)
				{
					std::cout << " forward rate extraction steps  test failed";
				}
				else
					--result;
		}


		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		thrust::device_vector<float> genFlows1_device(steps_to_test*n_vectors);
		thrust::device_vector<float> genFlows2_device(steps_to_test*n_vectors);
		std::vector<float> genFlows1_gpu_vec(genFlows1_device.size()); 
		std::vector<float> genFlows2_gpu_vec(genFlows2_device.size()); 

		{

			std::vector<float> genFlows1_vec(steps_to_test*n_vectors);
			std::vector<float> genFlows2_vec(steps_to_test*n_vectors);

			std::vector<float> aux_data_vec(2*steps_to_test+3);

			aux_data_vec[0] = 0.15f;
			aux_data_vec[1] = 0.1f;
			aux_data_vec[2] = 2.0f;

			for (int i=0; i < steps_to_test; ++i)
				aux_data_vec[3+i] =aux_data_vec[3+i+steps_to_test] =0.5f;

			thrust::device_vector<float> aux_data_device(    deviceVecFromStlVec(aux_data_vec));

			cudaThreadSynchronize();

			Timer h1;


			cashFlowGeneratorCallerTARN_main(genFlows1_device, 
				genFlows2_device, 
				aux_data_device, 
				n_vectors, 
				steps_to_test,
				extracted_forwards_device, 
				extracted_forwards_device, 
				extracted_forwards_device, 
				evolved_rates_device, 
				discounts_device);
			cudaThreadSynchronize();                                                 



			double time = h1.timePassed();
			std::cout << " time taken for cashFlowGeneratorCallerTARN_main computation."    << time << std::endl;


			Timer h3;


			cashFlowGeneratorCallerTARN_gold(  genFlows1_vec, 
				genFlows2_vec, 
				aux_data_vec, 
				n_vectors, 
				steps_to_test,
				extracted_forwards_cpu_vec, 
				extracted_forwards_cpu_vec, 
				extracted_forwards_cpu_vec, 
				evolved_rates_gpu_vec, 
				discounts_cpu_vec);

			double time3 =	 h3.timePassed();



			std::cout << " time taken for cashFlowGeneratorCallerTARN_gold on CPU: "    << time3 << std::endl;
			std::cout << " speed up " << time3/time << "\n";



			thrust::host_vector<float> genFlows1_host(genFlows1_device);
			std::copy(genFlows1_host.begin(),genFlows1_host.end(), genFlows1_gpu_vec.begin());

			thrust::host_vector<float> genFlows2_host(genFlows2_device);
			std::copy(genFlows2_host.begin(),genFlows2_host.end(), genFlows2_gpu_vec.begin());

			int count =0;

			double err1=0.0;
			double err2=0.0;
			for (int p = 0; p < n_vectors; ++p)
				for (int s=0; s < steps_to_test; ++s)
				{
					int i = p +     s*n_vectors;

					if (extraVerbose)
						std::cout << s << "," << p << "," << genFlows1_gpu_vec[i] << "," <<genFlows1_vec[i] << "\n";

					if (extraVerbose)
						std::cout << s << "," << p << "," << genFlows2_gpu_vec[i] << "," <<genFlows2_vec[i] << "\n";

					double erri =genFlows1_gpu_vec[i] - genFlows1_vec[i];
					++count;

					err1 += fabs(erri);
					err2 += fabs(erri*erri);


					erri =genFlows2_gpu_vec[i] - genFlows2_vec[i];
					++count;

					err1 += fabs(erri);
					err2 += fabs(erri*erri);



				}




				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}



				if (L1err > tolerance)
				{
					std::cout << " cash flow generation test failed";
				}
				else
					--result;


		}





		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////

		{

			cudaThreadSynchronize();

			std::vector<int> firstIndex_vec;
			std::vector<int> secondIndex_vec;
			std::vector<float>  thetas_vec;

			std::vector<float> paymentTimes_vec( steps_to_test);
			for (int i=0; i < steps_to_test; ++i)
				paymentTimes_vec[i] = rateTimes_vec[i+1];


			generateCashFlowIndicesAndWeights( firstIndex_vec, 
				secondIndex_vec,
				thetas_vec, 
				rateTimes_vec,
				paymentTimes_vec
				);

			thrust::device_vector<int> firstIndex_dev(deviceVecFromStlVec(firstIndex_vec));
			thrust::device_vector<int> secondIndex_dev(deviceVecFromStlVec(secondIndex_vec));
			thrust::device_vector<float> thetas_dev(deviceVecFromStlVec(thetas_vec));

			thrust::device_vector<float> discountedFlows_dev(steps_to_test*n_vectors); // output
			thrust::device_vector<float> summedDiscountedFlows_dev(n_vectors);

			for (int shared=0; shared < 3; ++shared)

			{
				bool useSharedForDiscounting = shared ==0;
				bool useTextures = shared != 1;

				Timer h1;


				cashFlowDiscounting_gpu_main(firstIndex_dev, 
					secondIndex_dev,
					thetas_dev, 
					discounts_device, 
					genFlows1_device, 
					numeraire_values_device,
					n_vectors, 
					steps_to_test, 
					useTextures,
					useSharedForDiscounting,
					discountedFlows_dev, // output
					summedDiscountedFlows_dev); // output


				cudaThreadSynchronize();                                                 



				double time = h1.timePassed();
				std::cout << " time taken for cashFlowDiscounting_gpu_main computation."    << time << std::endl;
				std::vector<float> discountedFlows_vec(steps_to_test*n_vectors); // output
				std::vector<float> summedDiscountedFlows_vec(n_vectors);


				Timer h2;

				cashFlowDiscounting_gold<float>( firstIndex_vec, 
					secondIndex_vec,
					thetas_vec, 
					discounts_cpu_vec, 
					genFlows1_gpu_vec, 
					numeraire_values_vec,
					n_vectors, 
					steps_to_test, 
					discountedFlows_vec, // output
					summedDiscountedFlows_vec); // output



				double time2 = h2.timePassed();



				std::cout << " time taken for cashFlowDiscounting_gold on CPU: "    << time2 << std::endl;
				std::cout << " speed up " << time2/time << "\n";


				std::vector<float> discountedFlows_gpu_vec(discountedFlows_dev.size());
				thrust::host_vector<float> discountedFlows_host(discountedFlows_dev);
				std::copy(discountedFlows_host.begin(),discountedFlows_host.end(), discountedFlows_gpu_vec.begin());

				std::vector<float> summedDiscountedFlows_gpu_vec(summedDiscountedFlows_dev.size());
				thrust::host_vector<float> summedDiscountedFlows_host(summedDiscountedFlows_dev);
				std::copy(summedDiscountedFlows_host.begin(),summedDiscountedFlows_host.end(), summedDiscountedFlows_gpu_vec.begin());

				int count =0;

				double err1=0.0;
				double err2=0.0;


				for (int p = 0; p < n_vectors; ++p)
				{               
					if (extraVerbose)
						std::cout << p << "," << summedDiscountedFlows_gpu_vec[p] << "," <<summedDiscountedFlows_vec[p] << "\n";


					double erri =summedDiscountedFlows_gpu_vec[p] - summedDiscountedFlows_vec[p];
					++count;


					err1 += fabs(erri);
					err2 += fabs(erri*erri);



				}




				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}



				if (L1err > tolerance)
				{
					std::cout << " cash flow discounting test failed";
				}
				else
					--result;


			}

		}



		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////
		///////////////////////////////////////////////



	}

	cudaThreadExit();


	return result;
}


int LMMPCTestRoutine(bool verbose, bool useTextures, DeviceChooser& chooser)
{
	int path_offset=0;

	int result =1; 
	{
		if (verbose)
		{
			std::cout << "\n\nTesting LMM PC evolver routine ";
			if (!useTextures)
				std::cout << " not ";
			std::cout << " using textures where there's a choice.\n";
		}
		bool extraVerbose = false;

		cudaSetDevice(chooser.WhichDevice());


		int n_vectors =  intPower(2,19);
		int n_poweroftwo =6;
		int factors = 5;
		int n_steps = intPower(2,n_poweroftwo);
		int tot_dimensions = n_steps*factors;
		int N= n_vectors*tot_dimensions; 
		int number_rates = 40;

		int steps_to_test  = 40; // must be less than or equal to n_steps


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



		std::vector<int> aliveIndex(n_steps);
		for (int i=0; i < n_steps; ++i)
			aliveIndex[i] = i;

		thrust::device_vector<float> evolved_rates_device(outN);

		std::vector<float> logRates_vec(number_rates);
		std::vector<float> rates_vec(number_rates);
		std::vector<float> taus_vec(number_rates);
		std::vector<float> displacements_vec(number_rates);
		for (int i=0; i < number_rates; ++i)
		{
			rates_vec[i] = 0.05f+0.01f*i;
			displacements_vec[i] =0.0f ; //0.02+0.001*i;
			logRates_vec[i] = logf(rates_vec[i]+displacements_vec[i] );
			taus_vec[i] = 0.5;

		}   

		thrust::host_vector<float> taus_host(taus_vec.begin(),taus_vec.end());
		thrust::device_vector<float> taus_device(taus_host);
		std::vector<float> evolved_rates_cpu_vec(n_vectors*number_rates*n_steps);

		thrust::host_vector<float> displacements_host(displacements_vec.begin(),displacements_vec.end());
		thrust::device_vector<float> displacements_device(displacements_host);



		std::vector<float> A(number_rates*factors*n_steps);

		{
			int i=0;
			for (int s=0; s < n_steps;++s)
				for (int r=0; r < number_rates; ++r)
					for (int f=0; f< factors; ++f, ++i)
						if (r >= aliveIndex[s])
							A[i] = static_cast<float>((A.size()+i+1.0f)/A.size())/(5.0f*factors);
						else
							A[i] =0.0f;
		}

		CubeConstFacade<float> pseudos(&A[0],n_steps,number_rates,factors);


		thrust::host_vector<float> A_host(A.begin(),A.end());
		thrust::device_vector<float> A_device(A_host.size());
		A_device  = A_host;

		thrust::host_vector<int> alive_host(aliveIndex.begin(),aliveIndex.end());
		thrust::device_vector<int> alive_device(alive_host);





		std::vector<float> drifts(n_steps*number_rates);
		MatrixFacade<float> drifts_mat(&drifts[0],n_steps,number_rates);
		for (int i=0; i < static_cast<int>(drifts.size()); ++i)
		{
			for (int s=0; s < n_steps; ++s)
				for (int r=0; r < number_rates; ++r)
				{  
					float x = 0.0f;
					for (int f=0; f< factors; ++f)
						x += pseudos(s,r,f)*pseudos(s,r,f);

					drifts_mat(s,r) = -0.5f*x;

				}
		}


		thrust::host_vector<float> drifts_host(drifts.begin(),drifts.end());
		thrust::device_vector<float> drifts_device(drifts_host);



		thrust::host_vector<float> logRates_host(logRates_vec.begin(),logRates_vec.end());
		thrust::device_vector<float> logRates_device(logRates_host);

		thrust::host_vector<float> rates_host(rates_vec.begin(),rates_vec.end());
		thrust::device_vector<float> rates_device(rates_host);


		thrust::device_vector<float> dev_output(N);    
		thrust::device_vector<float> device_input(N);

		thrust::device_vector<float> e_buffer_device(factors*n_vectors);
		thrust::device_vector<float> e_buffer_2_device(factors*n_vectors);

		thrust::device_vector<float> initial_drifts_device(number_rates);

		{

			spotDriftComputer<float> driftComp(taus_vec,  factors,displacements_vec);
			std::vector<float> initial_drifts_vec(number_rates);

			driftComp.getDrifts(pseudos[0], 
				rates_vec,
				initial_drifts_vec);

			thrust::host_vector<float> initialDrifts_host(initial_drifts_vec.begin(),initial_drifts_vec.end());
			initial_drifts_device = initialDrifts_host;

		}





		cudaThreadSynchronize();

		Timer h1;


		bool doNormalInSobol=true;
		int res =  SobDevice( n_vectors, tot_dimensions, path_offset,device_input,doNormalInSobol);
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




		thrust::device_vector<float> dev_correlated_rates(outN);

		thrust::device_vector<float> evolved_log_rates_device(outN);


		if (extraVerbose)
		{
			std::cout << "logRates_device " << "\n";

			for (int i=0; i < number_rates; ++i)
				std::cout << logRates_device[i] << ",";

			std::cout << "\n ";
			std::cout << "rates_device ";

			for (int i=0; i < number_rates; ++i)
				std::cout << rates_device[i] << ",";

			std::cout << "\n taus device  ";
			for (int i=0; i < number_rates; ++i)
				std::cout << taus_device[i] << ",";

			std::cout << "\n A device \n";
			for (int s=0; s < n_steps; ++s)
			{
				std::cout << s << "\n";
				for (int r=0; r < number_rates; ++r)
				{
					for (int f=0; f < factors; ++f)
						std::cout << A_device[s*number_rates*factors+r*factors+f]<<",";

					std::cout << "\n";
				}

			}
			std::cout << "\n Initial Drifts device  ";

			for (int i=0; i < number_rates; ++i)
				std::cout << initial_drifts_device[i] << ",";

			std::cout << "\n";
		}





		correlated_drift_paths_device( dev_output,
			dev_correlated_rates, // correlated rate increments 
			A_device, // correlator 
			alive_device,
			drifts_device,
			factors*number_rates,
			factors, 
			number_rates,
			n_vectors,
			n_steps);

		LMM_evolver_pc_main(  rates_device, 
			logRates_device, 
			taus_device,
			dev_correlated_rates,
			A_device,
			initial_drifts_device,
			displacements_device,
			aliveIndex,
			n_vectors,
			factors,
			steps_to_test, 
			number_rates, 
			e_buffer_device,
			e_buffer_2_device,
			evolved_rates_device, // for output
			evolved_log_rates_device  // for output 
			);


		double time = h1.timePassed();
		std::cout << " time taken for all steps of path generation."    << time << std::endl;

		thrust::host_vector<float> evolved_rates_host(evolved_rates_device);
		thrust::host_vector<float> evolved_log_rates_host(evolved_log_rates_device);

		std::vector<float> evolved_rates_gpu_vec(evolved_rates_host.begin(),evolved_rates_host.end());
		std::vector<float> evolved_log_rates_gpu_vec(evolved_log_rates_host.begin(), evolved_log_rates_host.end());

		std::vector<float> evolved_log_rates_cpu_vec(n_vectors*number_rates*n_steps);



		thrust::host_vector<float> rateIncrements_host(dev_correlated_rates);
		std::vector<float> rateIncrements_vec(rateIncrements_host.begin(),rateIncrements_host.end());



		int t0=clock();

		LMM_evolver_pc_gold(rates_vec,
			taus_vec,
			rateIncrements_vec, //  AZ  + mu_fixed
			A,
			displacements_vec,
			n_vectors,
			factors,
			steps_to_test, 
			number_rates, 
			evolved_rates_cpu_vec,
			evolved_log_rates_cpu_vec);

		int t1=clock();

		float time2 = (t1-t0+0.0f)/CLOCKS_PER_SEC;

		std::cout << " time taken for CPU evolution " << time2 << "\n";

		std::cout << " speed up ratio " << time2/time << "\n";



		double err1=0.0;
		double err2 = 0.0;

		int count =0;

		for (int p = 0; p < n_vectors; ++p)
			for (int s=0; s < steps_to_test; ++s)
			{
				if (extraVerbose)
				{   
					std::cout << p << "," << s ;
					for (int r=0; r< aliveIndex[s];  ++r)     
						std::cout <<  ",";

				}



				for (int r=aliveIndex[s]; r < number_rates; ++r)     
				{
					int i = p +    n_vectors*(r+ s*number_rates);
					if (extraVerbose)
						std::cout <<  "," << evolved_rates_gpu_vec[i] ;


					double erri =2* (evolved_rates_gpu_vec[i] - evolved_rates_cpu_vec[i])/(evolved_rates_gpu_vec[i] + evolved_rates_cpu_vec[i]);
					++count;
					err1 += fabs(erri);
					err2 += fabs(erri*erri);
				}

				if (extraVerbose)
				{

					std::cout << ",,";

					for (int r=0; r < number_rates; ++r)     
					{
						int i = p +    n_vectors*(r+ s*number_rates);
						std::cout <<  "," << evolved_rates_cpu_vec[i] ;
					}

					std::cout << ",,";

					for (int r=0; r < number_rates; ++r)     
					{
						int i = p +    n_vectors*(r+ s*number_rates);
						std::cout <<  "," << evolved_log_rates_gpu_vec[i] ;
					}

					std::cout << ",,";

					for (int r=0; r < number_rates; ++r)     
					{
						int i = p +    n_vectors*(r+ s*number_rates);
						std::cout <<  "," << evolved_log_rates_cpu_vec[i] ;
					}



					std::cout << "\n";
				}
			}




			double L1err = err1/count;

			double L2err = sqrt(err2/count);


			if (verbose)
			{
				std::cout << "L1  error " << L1err << "\n";
				std::cout << "L2  error " << L2err << "\n";
			}

			if (L1err > tolerance)
			{
				std::cout << " LMM PC Evolver  test failed";
			}
			else
				--result;
	}



	cudaThreadExit();


	return result;
}


	int LMMPCSKTestRoutine(bool verbose, int firstj, int lastj, int firstThreads, int lastThread, int threadstep, DeviceChooser& chooser)
{
	int path_offset=0;

	int result =lastj-firstj; 
	for (int j=firstj; j < lastj; ++j)
	{
		bool useSharedMem = (j ==3);
		bool doDiscounts =  (j==1);

		bool kepler = (j==0);

		if (verbose)
		{
			std::cout << "\n\nTesting LMM PC SK evolver routine. UseSharedMem=  " << useSharedMem << "   doDiscounts= "<< doDiscounts << "\n";
			std::cout << " \n kepler = " << kepler << "\n";
		}
		float cutoffLevel = 100.0;

		bool extraVerbose = false;

		cudaSetDevice(chooser.WhichDevice());
		{ // create scope to make sure all destroyed before thread exit

			int n_vectors =   intPower(2,19);
			int n_poweroftwo =5;
			int factors = 5;
			int n_steps = intPower(2,n_poweroftwo);
			int tot_dimensions = n_steps*factors;
			int N= n_vectors*tot_dimensions; 
			int number_rates = n_steps;

			int steps_to_test  = n_steps; // must be less than or equal to n_steps


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

			std::vector<int> aliveIndex(n_steps);
			for (int i=0; i < n_steps; ++i)
				aliveIndex[i] = i;

			thrust::device_vector<float> evolved_rates_device(outN);
			thrust::device_vector<float> discounts_device(n_steps*(number_rates+1)*n_vectors);

			std::vector<float> logRates_vec(number_rates);
			std::vector<float> rates_vec(number_rates);
			std::vector<float> taus_vec(number_rates);
			std::vector<float> displacements_vec(number_rates);
			for (int i=0; i < number_rates; ++i)
			{
				rates_vec[i] = 0.05f+0.01f*i;
				displacements_vec[i] =0.0f ; //0.02+0.001*i;
				logRates_vec[i] = logf(rates_vec[i]+displacements_vec[i] );
				taus_vec[i] = 0.5;

			}   

			thrust::host_vector<float> taus_host(taus_vec.begin(),taus_vec.end());
			thrust::device_vector<float> taus_device(taus_host);
			std::vector<float> evolved_rates_cpu_vec(n_vectors*number_rates*n_steps);

			thrust::host_vector<float> displacements_host(displacements_vec.begin(),displacements_vec.end());
			thrust::device_vector<float> displacements_device(displacements_host);



			std::vector<float> A(number_rates*factors*n_steps);

			{
				int i=0;
				for (int s=0; s < n_steps;++s)
					for (int r=0; r < number_rates; ++r)
						for (int f=0; f< factors; ++f, ++i)
							if (r >= aliveIndex[s])
								A[i] =static_cast<float>((A.size()+i+1.0f)/A.size())/(5.0f*factors);
							else
								A[i] =0.0f;
			}

			CubeConstFacade<float> pseudos(&A[0],n_steps,number_rates,factors);


			thrust::host_vector<float> A_host(A.begin(),A.end());
			thrust::device_vector<float> A_device(A_host.size());
			A_device  = A_host;

			thrust::host_vector<int> alive_host(aliveIndex.begin(),aliveIndex.end());
			thrust::device_vector<int> alive_device(alive_host);





			std::vector<float> drifts(n_steps*number_rates);
			MatrixFacade<float> drifts_mat(&drifts[0],n_steps,number_rates);
			for (int i=0; i < static_cast<int>(drifts.size()); ++i)
			{
				for (int s=0; s < n_steps; ++s)
					for (int r=0; r < number_rates; ++r)
					{  
						float x = 0.0f;
						for (int f=0; f< factors; ++f)
							x += pseudos(s,r,f)*pseudos(s,r,f);

						drifts_mat(s,r) = -0.5f*x;

					}
			}


			thrust::host_vector<float> drifts_host(drifts.begin(),drifts.end());
			thrust::device_vector<float> drifts_device(drifts_host);

			thrust::host_vector<float> logRates_host(logRates_vec.begin(),logRates_vec.end());
			thrust::device_vector<float> logRates_device(logRates_host);

			thrust::host_vector<float> rates_host(rates_vec.begin(),rates_vec.end());
			thrust::device_vector<float> rates_device(rates_host);


			thrust::device_vector<float> dev_output(N);    
			thrust::device_vector<float> device_input(N);

			thrust::device_vector<float> e_buffer_device(factors*n_vectors);
			thrust::device_vector<float> e_buffer_2_device(factors*n_vectors);

			thrust::device_vector<float> initial_drifts_device(number_rates);

			{

				spotDriftComputer<float> driftComp(taus_vec,  factors,displacements_vec);
				std::vector<float> initial_drifts_vec(number_rates);

				driftComp.getDrifts(pseudos[0], 
					rates_vec,
					initial_drifts_vec);

				thrust::host_vector<float> initialDrifts_host(initial_drifts_vec.begin(),initial_drifts_vec.end());
				initial_drifts_device = initialDrifts_host;

			}

			thrust::device_vector<float> dev_correlated_rates(outN);

			thrust::device_vector<float> evolved_log_rates_device(outN);



			double time;
			for (int threads= firstThreads; threads<=lastThread; threads+=threadstep)
			{



				bool doNormalInSobol=true;
				int res =  SobDevice( n_vectors, tot_dimensions, path_offset,device_input,doNormalInSobol);
				cudaThreadSynchronize();




				if (extraVerbose)
				{
					thrust::host_vector<float> bridgedNumbersHost(N);   
					bridgedNumbersHost = device_input;

					for (int i=0; i < N; ++i)
						std::cout << i << "," << bridgedNumbersHost[i]  << "\n";

				}


				BrownianBridgeMultiDim<float>::ordering allocator( BrownianBridgeMultiDim<float>::triangular);

				bool useTextures = true;

				MultiDBridge(n_vectors, 
					n_poweroftwo,
					factors,
					device_input, 
					dev_output,
					allocator,
					useTextures)
					;



				if (extraVerbose)
				{
					std::cout << "logRates_device " << "\n";

					for (int i=0; i < number_rates; ++i)
						std::cout << logRates_device[i] << ",";

					std::cout << "\n ";
					std::cout << "rates_device ";

					for (int i=0; i < number_rates; ++i)
						std::cout << rates_device[i] << ",";

					std::cout << "\n taus device  ";
					for (int i=0; i < number_rates; ++i)
						std::cout << taus_device[i] << ",";

					std::cout << "\n A device \n";
					for (int s=0; s < n_steps; ++s)
					{
						std::cout << s << "\n";
						for (int r=0; r < number_rates; ++r)
						{
							for (int f=0; f < factors; ++f)
								std::cout << A_device[s*number_rates*factors+r*factors+f]<<",";

							std::cout << "\n";
						}

					}
					std::cout << "\n Initial Drifts device  ";

					for (int i=0; i < number_rates; ++i)
						std::cout << initial_drifts_device[i] << ",";

					std::cout << "\n";
				}

				bool doTranspose=false;

				//		cudaStream_t streamNumber=0;
				cudaThreadSynchronize();

				Timer h1;


				if (kepler)
				{
					LMM_evolver_pc_single_kernel_discounts_kepler_main(  rates_device, 
						logRates_device, 
						taus_device, 
						dev_output,
						A_device,
						initial_drifts_device, 
						drifts_device, 
						displacements_device, 
						alive_device,
						n_vectors,
						factors,
						n_steps, 
						number_rates, 
						e_buffer_device,
						e_buffer_2_device,
						evolved_rates_device,
						evolved_log_rates_device,
						discounts_device, 
						doTranspose,
						threads
						);

					//		cutoffLevel =20.0;

				}
				else
					if (doDiscounts)
						LMM_evolver_pc_single_kernel_discounts_main(  rates_device, 
						logRates_device, 
						taus_device, 
						dev_output,
						A_device,
						initial_drifts_device, 
						drifts_device, 
						displacements_device, 
						alive_device,
						n_vectors,
						factors,
						n_steps, 
						number_rates, 
						e_buffer_device,
						e_buffer_2_device,
						evolved_rates_device,
						evolved_log_rates_device,
						discounts_device, 
						useSharedMem,
						doDiscounts,
						doTranspose,
						threads
						);

					else
						LMM_evolver_pc_single_kernel_main(  rates_device, 
						logRates_device, 
						taus_device, 
						dev_output,
						A_device,
						initial_drifts_device, 
						drifts_device, 
						displacements_device, 
						alive_device,
						n_vectors,
						factors,
						n_steps, 
						number_rates, 
						e_buffer_device,
						e_buffer_2_device,
						evolved_rates_device,
						evolved_log_rates_device,
						useSharedMem
						);







				time = h1.timePassed();
				std::cout << " time taken for turning normals into paths ,"    << time << std::endl;
			}
			thrust::host_vector<float> evolved_rates_host(evolved_rates_device);
			thrust::host_vector<float> evolved_log_rates_host(evolved_log_rates_device);

			std::vector<float> evolved_rates_gpu_vec(evolved_rates_host.begin(),evolved_rates_host.end());
			std::vector<float> evolved_log_rates_gpu_vec(evolved_log_rates_host.begin(), evolved_log_rates_host.end());

			std::vector<float> evolved_log_rates_cpu_vec(n_vectors*number_rates*n_steps);
			/*
			correlated_drift_paths_device( dev_output,
			dev_correlated_rates, // correlated rate increments 
			A_device, // correlator 
			alive_device,
			drifts_device,
			factors*number_rates,
			factors, 
			number_rates,
			n_vectors,
			n_steps);
			*/

			std::vector<float> uncorrelatedNumbersVec(stlVecFromDevVec( dev_output));

			std::vector<float> correlatedNumbersVec(outN);
			std::vector<float> discounts_cpu_vec(steps_to_test*n_vectors*(number_rates+1));
			std::vector<float> discounts_gpu_vec(steps_to_test*n_vectors*(number_rates+1));


			int t0=clock();

			correlate_drift_paths_gold(uncorrelatedNumbersVec, // randon numbers
				correlatedNumbersVec, // correlated rate increments 
				A, // correlator 
				number_rates*factors,
				drifts, // drifts 
				factors, 
				number_rates,
				n_vectors,
				n_steps);



			//	thrust::host_vector<float> rateIncrements_host(dev_correlated_rates);
			//			std::vector<float> rateIncrements_vec(rateIncrements_host.begin(),rateIncrements_host.end());




			LMM_evolver_pc_gold(rates_vec,
				taus_vec,
				correlatedNumbersVec, //  AZ  + mu_fixed
				A,
				displacements_vec,
				n_vectors,
				factors,
				steps_to_test, 
				number_rates, 
				evolved_rates_cpu_vec,
				evolved_log_rates_cpu_vec,
				cutoffLevel);

			if (doDiscounts)
				discount_ratios_computation_gold( evolved_rates_cpu_vec, 
				taus_vec, 
				aliveIndex, 
				n_vectors,
				steps_to_test, 
				number_rates, 
				discounts_cpu_vec  // for output 
				);                                                              



			int t1=clock();

			float time2 = (t1-t0+0.0f)/CLOCKS_PER_SEC;

			std::cout << " time taken for CPU evolution " << time2 << "\n";

			std::cout << " speed up ratio " << time2/time << "\n";
			if (doDiscounts)
			{
				thrust::host_vector<float> discounts_host(discounts_device);
				std::copy(discounts_host.begin(),discounts_host.end(),discounts_gpu_vec.begin()); 

			}


			double err1=0.0;
			double err2 = 0.0;

			int count =0;

			for (int p = 0; p < n_vectors; ++p)
				for (int s=0; s < steps_to_test; ++s)
				{
					if (extraVerbose)
					{   
						std::cout << p << "," << s ;
						for (int r=0; r< aliveIndex[s];  ++r)     
							std::cout <<  ",";

					}



					for (int r=aliveIndex[s]; r < number_rates; ++r)     
					{
						int i = p +    n_vectors*(r+ s*number_rates);
						if (extraVerbose)
							std::cout <<  "," << evolved_rates_gpu_vec[i] ;


						double erri =2* (evolved_rates_gpu_vec[i] - evolved_rates_cpu_vec[i])/(evolved_rates_gpu_vec[i] + evolved_rates_cpu_vec[i]);
						++count;
						err1 += fabs(erri);
						err2 += fabs(erri*erri);
					}


					if (doDiscounts)
					{
						for (int r=aliveIndex[s]; r <= number_rates; ++r)     
						{
							int i = p +    n_vectors*(r+ s*(number_rates+1));

							float dg = discounts_gpu_vec[i];
							float dc = discounts_cpu_vec[i];


							double erri =2.0f* (dc-dg )/(dc+dg);
							++count;
							err1 += fabs(erri);
							err2 += fabs(erri*erri);
						}


					}


					if (extraVerbose)
					{

						std::cout << ",,";

						for (int r=0; r < number_rates; ++r)     
						{
							int i = p +    n_vectors*(r+ s*number_rates);
							std::cout <<  "," << evolved_rates_cpu_vec[i] ;
						}




						std::cout << "\n";
					}
				}

				if (doDiscounts && extraVerbose)
				{
					debugDumpCube(discounts_gpu_vec, "discounts_gpu_vec" ,  steps_to_test, number_rates+1,n_vectors );
				}


				double L1err = err1/count;

				double L2err = sqrt(err2/count);


				if (verbose)
				{
					std::cout << "L1  error " << L1err << "\n";
					std::cout << "L2  error " << L2err << "\n";
				}

				if (L1err > tolerance)
				{
					std::cout << " LMM PC Single Kernel Evolver  test failed, doDiscounts: " << doDiscounts << " sharedmem " << useSharedMem << "\n";
				}
				else
				{
					std::cout << " LMM PC Single Kernel Evolver  test passed, doDiscounts: " << doDiscounts << " sharedmem " << useSharedMem << "\n";
					--result;
				}
		}


		cudaThreadExit();

	}

	return result;
}
