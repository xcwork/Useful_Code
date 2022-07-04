//
//
//       LS_Test.cu
//
//
// (c) Mark Joshi 2011,2012,2013
// This code is released under the GNU public licence version 3

// routine to test the LS Code

#include <LS_test.h>
#include <gold/MultiLS_regression.h>
#include <gold\Regression_Selector_concrete_gold.h>
#include <LS_Basis_main.h>
#include <gold/LS_Basis_gold.h>
#include <cutil_inline.h>
#include <cutil.h>

#include <gold/Bridge_gold.h>
#include <LMM_evolver_main.h>
#include <multid_path_gen_BS_main.h>
#include <gold/LMM_evolver_gold.h>
#include <early_exercise_value_generator_main.h>
#include <gold/early_exercise_value_generator_gold.h>
#include <cashFlowDiscounting_main.h>
#include <gold/cashFlowDiscounting_gold.h>
#include <gold/ExerciseIndices_gold.h>
#include <cashFlowGeneration_product_main.h>

#include<cashFlowGen_earlyEx_product_main.h>
#include <gold/cashFlowGeneration_product_gold.h>
#include <gold/cashFlowDiscounting_gold.h>
#include <gold/cashFlowAggregation_gold.h>
#include <cashFlowAggregation_main.h>

#include <gold/LS_regression.h>
#include <LS_main.h>

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
#include <InverseCumulatives.h>
#include <gold/Bridge_gold.h>
#include <sobol.h>
#include <Bridge.h>
#include <correlate_drift_main.h>
#include <cudaMacros.h>
#include <gold/MatrixFacade.h> 

#include <gold/early_exercise_value_generator_gold.h>
#include <gold/cashFlowGeneration_earlyEx_product.h>
#include <Utilities.h>
#include <ComparisonCheck.h>

#include <gold/math/fp_utilities.h>
#include <gold/math/pseudoSquareRoot.h>
#include <gold/volstructs_gold.h>
#include <LS_main_cula.h>

#include <numeric>
#include <RegressionSelectorConcrete.h>
namespace
{
	double tolerance = 1E-6;
	double misMatchTol = 1E-4;

	double LStolerance = 2E-4;
}


int Test_Basis_Variable_Extraction_etc(bool verbose)
{
	int fails=0; 
	for (int useCula=0; useCula <2; useCula++)
	{
		bool culaBool = (useCula == 0);
		for (int useCrossTerms=0; useCrossTerms <2; useCrossTerms++)
			for (int useLogBasis =0; useLogBasis <2; useLogBasis++)
			{
				if (useLogBasis==1 && !culaBool)
					for (int normalise=0; normalise <2; ++normalise)
						fails +=Test_LS_Code( verbose, useLogBasis>0, normalise>0, useCrossTerms>0, culaBool);
				else
					fails +=Test_LS_Code( verbose, useLogBasis>0, false, useCrossTerms>0,culaBool );
		
			}
	}
		
	return fails;

}

int Test_LS_Code(bool verbose, bool useLogBasis, bool normalise, bool useCrossTerms, bool useCula)
{

	if (verbose)
	{
		std::cout << "\n\nTesting Test_Basis_Variable_Extraction etc routine ";           
	}
	int numberFails =10;

		int pathsPerBatch = 32767;
		int lowerPathCutoff =1000;

		int mb=5;
		int maxb=6;

#ifdef _DEBUG
		pathsPerBatch = 255;
		lowerPathCutoff = 10;
	     mb = 1;
		 maxb=2;
#endif

	    int numberOfExtraRegressions = 3;
		int maxRegressionDepth = numberOfExtraRegressions+1;



		double lowerFrac = 0.49;
		double upperFrac=0.51;
		double initialSDguess = 2;
		double multiplier =0.8;
		RegressionSelector_gold_Fraction regressionSelector(lowerFrac, upperFrac, initialSDguess,  multiplier);


	for(	int numberOfBatches = mb; numberOfBatches < maxb;++numberOfBatches)
	{
		numberFails =10;	
	
		std::cout << "using " << numberOfBatches << " of " << pathsPerBatch << " for first pass\n";

		int numberOfSecondPassBatches = numberOfBatches;
		int pathsPerSecondPassBatch = pathsPerBatch; 
		int duplicate=1; // if zero use same paths as for first pass, if 1 start at the next paths



		int totalPaths = pathsPerBatch*numberOfBatches;

		int n_vectors =  pathsPerBatch;
		int n_poweroftwo =2;

		int numberSteps = intPower(2,n_poweroftwo);
		int number_rates = numberSteps;
		int factors = min(number_rates,5);

		//float  beta =0.2f;
		double beta_d = 0.2;
		//	float  L =0.0f;
		double L_d = 0.0;

		int tot_dimensions = numberSteps*factors;
		int N= n_vectors*tot_dimensions; 

		int numberOfRates = number_rates;
		int numberOfSteps = numberSteps;

		int basisVariablesPerStep = basisVariableExample_gold::maxVariablesPerStep();


		bool useTextures = true;

		std::cout << " uselog basis," << useLogBasis << ", normalise , " << normalise << ", useCrossTerms , " << useCrossTerms << ", cula " << useCula << "\n";

		double totalTimeForGPUExtraction=0.0;
		double totalTimeForCPUExtraction=0.0;

		double totalTimeForSpotMeasureGPU =0.0;
		double totalTimeForSpotMeasureCPU =0.0;

		double totalTimeForExerciseValuesGPU =0.0;
		double totalTimeForExerciseValuesCPU =0.0;

		double totalTimeForCashFlowGenerationGPU=0.0;
		double totalTimeForCashFlowGenerationCPU=0.0;

		double totalTimeForCashFlowPartialDiscountingCPU=0.0;
		double totalTimeForCashFlowPartialDiscountingGPU=0.0;

		double totalTimeForCashFlowAggregationCPU=0.0;
		double totalTimeForCashFlowAggregationGPU=0.0;


		int  cashFlow1Mismatches=0; 
		int  cashFlow2Mismatches=0;

		int cashFlowDiscounting1_partial_mismatches=0;
		int cashFlowDiscounting2_partial_mismatches=0;

		bool useSharedForDiscounting=true;
		bool useTexturesForDiscounting = true;



		cudaSetDevice(cutGetMaxGflopsDeviceId());
		cudaThreadSynchronize();

		// set up all device vectors first
		// let's put everything inside a little scope so it all destroys earlier
		{
			std::vector<int> exerciseIndices_vec; // the indices of the exercise times amongst the evolution times
			std::vector<int> exerciseIndicators_vec(numberSteps);
			for (int i=0; i < numberSteps;++i) // 
			{
				exerciseIndicators_vec[i] =1;
			}

			//   exerciseIndicators_vec[1] =0;


			GenerateIndices(exerciseIndicators_vec, exerciseIndices_vec);

			int numberOfExerciseDates = exerciseIndices_vec.size();

			int totalNumberOfBasisVariables = basisVariablesPerStep*numberOfExerciseDates*totalPaths;

			// across batch data
			thrust::device_vector<float> final_basis_variables_device(totalNumberOfBasisVariables);
			std::vector<float> final_basis_variables_vec(totalNumberOfBasisVariables);

			CubeFacade<float> basisFunctionVariables_cube(&final_basis_variables_vec[0],exerciseIndices_vec.size(),basisVariablesPerStep,totalPaths);

			thrust::device_vector<float> spot_measure_values_device(numberOfSteps*totalPaths);
			std::vector<float> spot_measure_values_vec(numberOfSteps*totalPaths);
			MatrixFacade<float>  spot_measure_values_matrix(&spot_measure_values_vec[0],numberOfSteps,totalPaths);




			thrust::device_vector<int> alive_device;   
			int steps_to_test  = numberSteps; // must be less than or equal to numberSteps


			thrust::host_vector<int> exerciseIndices_host(exerciseIndices_vec.begin(),exerciseIndices_vec.end());
			thrust::host_vector<int> exerciseIndicators_host(exerciseIndicators_vec.begin(),exerciseIndicators_vec.end());


			thrust::device_vector<int> exerciseIndices_device(exerciseIndices_host);
			thrust::device_vector<int> exerciseIndicators_device(exerciseIndicators_host);

			int numberExerciseDates =  exerciseIndices_vec.size();
			int totalNumberOfExerciseValues =numberExerciseDates*totalPaths;
			thrust::device_vector<float> exercise_values_device(totalNumberOfExerciseValues);

			//     for (int i=0; i < totalNumberOfExerciseValues; ++i)
			//      exercise_values_device[i] = i;

			std::vector<float> exercise_values_vec(totalNumberOfExerciseValues);
			MatrixFacade<float>  exercise_values_matrix(&exercise_values_vec[0],exerciseIndices_vec.size(),totalPaths);


			int outN = n_vectors*numberSteps*number_rates;

			std::vector<int> deflation_locations_vec(numberSteps);
			std::vector<int> aliveIndex(numberSteps);
			for (int i=0; i < numberSteps; ++i)
			{
				aliveIndex[i] = i;
				deflation_locations_vec[i]=i;
			}

			thrust::device_vector<int> deflation_locations_device(deflation_locations_vec.begin(),deflation_locations_vec.end());

			thrust::device_vector<float> evolved_rates_device(outN);

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

			thrust::device_vector<float> aggregatedFlows_device(totalPaths*exerciseIndices_vec.size());




			std::vector<float> aggregatedFlows_vec(totalPaths*exerciseIndices_vec.size());
			MatrixFacade<float> aggregatedFlows_cpu_matrix(&aggregatedFlows_vec[0],exerciseIndices_vec.size(),totalPaths);


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

				std::vector<float> drifts(numberSteps*number_rates);
				MatrixFacade<float> drifts_mat(&drifts[0],numberSteps,number_rates);
				for (int i=0; i < static_cast<int>(drifts.size()); ++i)
				{
					for (int s=0; s < numberSteps; ++s)
						for (int r=0; r < number_rates; ++r)
						{  
							float x = 0.0f;
							for (int f=0; f< factors; ++f)
								x += pseudos(s,r,f)*pseudos(s,r,f);

							drifts_mat(s,r) = -0.5f*x;

						}
				}

				thrust::device_vector<float> drifts_device(deviceVecFromStlVec(drifts));
				thrust::device_vector<float> logRates_device(deviceVecFromStlVec(logRates_vec));
				thrust::device_vector<float> rates_device(deviceVecFromStlVec(rates_vec));
				thrust::device_vector<float> dev_output(N);    
				thrust::device_vector<float> device_input(N);
				thrust::device_vector<float> e_buffer_device(factors*n_vectors);
				thrust::device_vector<float> initial_drifts_device(number_rates);

				{

					spotDriftComputer driftComp(taus_vec,  factors,displacements_vec);
					std::vector<float> initial_drifts_vec(number_rates);

					driftComp.getDrifts(pseudos[0], 
						rates_vec,
						initial_drifts_vec);

					thrust::host_vector<float> initialDrifts_host(initial_drifts_vec.begin(),initial_drifts_vec.end());
					initial_drifts_device = initialDrifts_host;

				}
				// rates for exercise values
				thrust::device_vector<float>  rate1_device(n_vectors*numberOfExerciseDates); // one rate per exercise date per path
				thrust::device_vector<float>  rate2_device(n_vectors*numberOfExerciseDates);
				thrust::device_vector<float>  rate3_device(n_vectors*numberOfExerciseDates);

				thrust::host_vector<float> rate1_host(n_vectors*numberOfExerciseDates);
				thrust::host_vector<float> rate2_host(n_vectors*numberOfExerciseDates);
				thrust::host_vector<float> rate3_host(n_vectors*numberOfExerciseDates);

				MatrixConstFacade<float> rate1_matrix(&rate1_host[0],
					numberOfExerciseDates,
					pathsPerBatch);

				MatrixConstFacade<float> rate2_matrix(&rate2_host[0],
					numberOfExerciseDates,
					pathsPerBatch);

				MatrixConstFacade<float> rate3_matrix(&rate3_host[0],
					numberOfExerciseDates,
					pathsPerBatch);

				std::vector<float> rate1_vec(n_vectors*numberOfExerciseDates);
				std::vector<float> rate2_vec(n_vectors*numberOfExerciseDates);
				std::vector<float> rate3_vec(n_vectors*numberOfExerciseDates);


				std::vector<int> extractor1(numberOfExerciseDates);
				for (int i=0; i < static_cast<int>(extractor1.size()); ++i)
					extractor1[i] = exerciseIndices_vec[i];

				std::vector<int> extractor2(numberOfExerciseDates);
				for (int i=0; i < static_cast<int>(extractor2.size()); ++i)
					extractor2[i] = std::min( exerciseIndices_vec[i]+1,steps_to_test-1);

				std::vector<int> extractor3(numberOfExerciseDates);
				for (int i=0; i < static_cast<int>(extractor3.size()); ++i)
					extractor3[i] = steps_to_test-1;

				// we want co-terminal swap-rates and their annuities and their swap-rates for Berm test
				thrust::device_vector<float> annuities_device(n_vectors*steps_to_test*number_rates);
				thrust::device_vector<float> swap_rates_device(n_vectors*steps_to_test*number_rates);

				// rates for cash-flow generations
				thrust::device_vector<float>  rate1_cf_device(n_vectors*steps_to_test); // one rate per step per path
				thrust::device_vector<float>  rate2_cf_device(n_vectors*steps_to_test);
				thrust::device_vector<float>  rate3_cf_device(n_vectors*steps_to_test);

				thrust::host_vector<float> rate1_cf_host(n_vectors*steps_to_test);
				thrust::host_vector<float> rate2_cf_host(n_vectors*steps_to_test);
				thrust::host_vector<float> rate3_cf_host(n_vectors*steps_to_test);

				MatrixConstFacade<float> rate1_cf_matrix(&rate1_cf_host[0],
					steps_to_test,
					pathsPerBatch);

				MatrixConstFacade<float> rate2_cf_matrix(&rate2_cf_host[0],
					steps_to_test,
					pathsPerBatch);

				MatrixConstFacade<float> rate3_cf_matrix(&rate3_cf_host[0],
					steps_to_test,
					pathsPerBatch);

				std::vector<float> rate1_cf_vec(n_vectors*steps_to_test);
				std::vector<float> rate2_cf_vec(n_vectors*steps_to_test);
				std::vector<float> rate3_cf_vec(n_vectors*steps_to_test);


				std::vector<int> extractor1_cf(steps_to_test);
				for (int i=0; i < static_cast<int>(extractor1_cf.size()); ++i)
					extractor1_cf[i] = i;

				std::vector<int> extractor2_cf(steps_to_test);
				for (int i=0; i < static_cast<int>(extractor2_cf.size()); ++i)
					extractor2_cf[i] = std::min( i+1,steps_to_test-1);

				std::vector<int> extractor3_cf(steps_to_test);
				for (int i=0; i < static_cast<int>(extractor3_cf.size()); ++i)
					extractor3_cf[i] =std::min( i+2,steps_to_test-1);


				thrust::device_vector<int> integer_Data_device(1); //  data specific to the basis variables
				thrust::device_vector<float> float_Data_device(3,maxDisplacement);
				float_Data_device[2] =0.0f;

				std::vector<int> integerData_vec(1);
				std::vector<float> floatData_vec(3,maxDisplacement);
				floatData_vec[2] = 0.0f;

				float strike = 0.05f;
				thrust::device_vector<int> integer_Data_evalue_device(1); //  data specific to the exercise value
				thrust::device_vector<float> float_Data_evalue_device(1); 
				std::vector<int> integer_Data_evalue_vec(1);
				std::vector<float> float_Data_evalue_vec(1);

				float_Data_evalue_device[0] = float_Data_evalue_vec[0] = strike;


				thrust::host_vector<float> evolved_rates_host(evolved_rates_device.size());
				thrust::device_vector<float> dev_correlated_rates(outN);

				thrust::device_vector<float> evolved_log_rates_device(outN);


				// discounting tests require us to have the target indices set up 
				// and payment times

				std::vector<int> genTimeIndex_vec(numberSteps);
				for (size_t i=0;i< genTimeIndex_vec.size(); ++i)
					genTimeIndex_vec[i] = i;

				thrust::device_vector<int> genTimeIndex_device(deviceVecFromStlVec(genTimeIndex_vec));

				std::vector<int> stepToExerciseIndices_vec;

				findExerciseTimeIndicesFromPaymentTimeIndices(genTimeIndex_vec, 
					exerciseIndices_vec,
					exerciseIndicators_vec, // redundant info but sometimes easier to work with
					stepToExerciseIndices_vec
					);
				thrust::host_vector<int> stepToExerciseIndices_host(stepToExerciseIndices_vec.begin(), stepToExerciseIndices_vec.end());
				thrust::device_vector<int> stepToExerciseIndices_device(stepToExerciseIndices_host);

				std::vector<float> paymentTimes_vec(numberSteps);
				for (size_t i=0;i< paymentTimes_vec.size(); ++i)
					paymentTimes_vec[i] = rateTimes_vec[i+1];

				std::vector<int> firstIndex_vec, secondIndex_vec;
				std::vector<float> thetas_vec;

				generateCashFlowIndicesAndWeights<float>( firstIndex_vec, 
					secondIndex_vec,
					thetas_vec,
					rateTimes_vec,
					paymentTimes_vec
					);

				thrust::device_vector<float> discounted_flows1_device(n_vectors*steps_to_test);
				std::vector<float> discounted_flows1_cpu_vec(n_vectors*steps_to_test);
				MatrixFacade<float> discounted_flows1_cpu_Matrix(&discounted_flows1_cpu_vec[0],steps_to_test,n_vectors);
				MatrixConstFacade<float> discounted_flows1_cpu_constMatrix(&discounted_flows1_cpu_vec[0],steps_to_test,n_vectors);

				thrust::device_vector<float> discounted_flows2_device(n_vectors*steps_to_test);
				std::vector<float> discounted_flows2_cpu_vec(n_vectors*steps_to_test);
				MatrixFacade<float> discounted_flows2_cpu_Matrix(&discounted_flows2_cpu_vec[0],steps_to_test,n_vectors);
				MatrixConstFacade<float> discounted_flows2_cpu_constMatrix(&discounted_flows2_cpu_vec[0],steps_to_test,n_vectors);


				thrust::device_vector<int> firstIndex_device(deviceVecFromStlVec(firstIndex_vec));
				thrust::device_vector<int> secondIndex_device(deviceVecFromStlVec(secondIndex_vec));
				thrust::device_vector<float> thetas_device(deviceVecFromStlVec(thetas_vec));




				// set up swap product data

				thrust::device_vector<float> genFlows1_device(n_vectors*steps_to_test);
				thrust::device_vector<float> genFlows2_device(genFlows1_device.size());
				std::vector<float> genFlows1_cpu_vec(genFlows1_device.size());
				std::vector<float> genFlows2_cpu_vec(genFlows2_device.size());
				MatrixFacade<float> genFlows1_cpu_matrix(&genFlows1_cpu_vec[0],steps_to_test,n_vectors);
				MatrixConstFacade<float> genFlows1_cpu_constmatrix(&genFlows1_cpu_vec[0],steps_to_test,n_vectors);

				MatrixFacade<float> genFlows2_cpu_matrix(&genFlows2_cpu_vec[0],steps_to_test,n_vectors);
				MatrixConstFacade<float> genFlows2_cpu_constmatrix(&genFlows2_cpu_vec[0],steps_to_test,n_vectors);


				std::vector<float> aux_swap_data_vec(2*number_rates+2);

				float payReceive = -1.0f;
				aux_swap_data_vec[0] = strike;
				aux_swap_data_vec[1] = payReceive;
				std::vector<float> accruals1(number_rates);
				std::vector<float> accruals2(number_rates);

				for (int j=0; j < number_rates; ++j)
				{
					aux_swap_data_vec[j+2] =0.5f;
					accruals1[j] = aux_swap_data_vec[j+2];
					aux_swap_data_vec[number_rates+j+2] =0.5f;
					accruals2[j] =  aux_swap_data_vec[number_rates+j+2];
				}

				thrust::device_vector<float> aux_swap_data_device(aux_swap_data_vec.begin(),aux_swap_data_vec.end());


				thrust::device_vector<float> spot_measure_values_small_device(n_vectors*steps_to_test);
				thrust::host_vector<float> spot_measure_values_small_host(n_vectors*steps_to_test);
				MatrixFacade<float> spot_measure_values_small_Matrix(&spot_measure_values_small_host[0],steps_to_test,n_vectors);
				MatrixConstFacade<float> spot_measure_values_small_constMatrix(&spot_measure_values_small_host[0],steps_to_test,n_vectors);


				// every is now set up so can do  loops 

				cudaThreadSynchronize();

				for (int batchCounter=0; batchCounter < numberOfBatches; ++batchCounter)
				{
					int path_offset= batchCounter * pathsPerBatch;

					bool doNormalInSobol=true;
					SobDevice( n_vectors, tot_dimensions,path_offset, device_input,doNormalInSobol);
					cudaThreadSynchronize();

					BrownianBridgeMultiDim::ordering allocator( BrownianBridgeMultiDim::triangular);


					MultiDBridge(n_vectors, 
						n_poweroftwo,
						factors,
						device_input, 
						dev_output,
						allocator,
						useTextures);

					correlated_drift_paths_device( dev_output,
						dev_correlated_rates, // correlated rate increments 
						A_device, // correlator 
						alive_device,
						drifts_device,
						factors*number_rates,
						factors, 
						number_rates,
						n_vectors,
						numberSteps);

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

					//     std::cout << "pass 1\n";
					//      DumpDeviceVector(evolved_rates_device);


					evolved_rates_host =evolved_rates_device;



					evolved_rates_gpu_vec.resize(evolved_rates_host.size());
					std::copy(evolved_rates_host.begin(),evolved_rates_host.end(),evolved_rates_gpu_vec.begin());
					cudaThreadSynchronize();

					//	debugDumpCube(evolved_rates_gpu_vec,"evolved rates", steps_to_test,number_rates,n_vectors);

					bool allStepsAtOnce =true;

					discount_ratios_computation_main( evolved_rates_device, 
						taus_device, 
						aliveIndex, 
						alive_device,
						n_vectors,
						steps_to_test, 
						number_rates, 
						discounts_device,
						allStepsAtOnce  
						);

					discounts_host=discounts_device;


					cudaThreadSynchronize();

					// set up rate vectors 

					coterminal_annuity_ratios_computation_gpu(  discounts_device, 
						taus_device, 
						aliveIndex, 
						n_vectors,
						steps_to_test, 
						number_rates, 
						annuities_device  // for output 
						);

					coterminal_swap_rates_computation_main_gpu(  discounts_device, 
						annuities_device, 
						aliveIndex, 
						n_vectors,
						steps_to_test, 
						number_rates, 
						swap_rates_device  // for output 
						);





					cudaThreadSynchronize();

					//       debugDumpCube(annuities_device,"annuities_device", steps_to_test,number_rates,n_vectors);
					//		debugDumpCube(swap_rates_device,"swap_rates_device", steps_to_test,number_rates,n_vectors);



					forward_rate_extraction_gpu_main(  evolved_rates_device, 
						extractor1,                          
						n_vectors,
						numberOfExerciseDates,
						number_rates, 
						rate1_device                
						);

					forward_rate_extraction_gpu_main(//evolved_rates_device, 
						swap_rates_device, 
						extractor2,                          
						n_vectors,
						numberOfExerciseDates,
						number_rates, 
						rate2_device                
						);

					forward_rate_extraction_gpu_main( // evolved_rates_device,
						annuities_device, 
						extractor3,                          
						n_vectors,
						numberOfExerciseDates,
						number_rates, 
						rate3_device                
						);

					cudaThreadSynchronize();


					//	debugDumpMatrix(rate2_device,"rate 2", steps_to_test,n_vectors);
					// now for cash flows

					/*      std::cout << "pass 1 rate 1\n ";
					DumpDeviceVector(rate1_device);


					std::cout << "pass 1 rate 2\n";
					DumpDeviceVector(rate2_device);

					std::cout << "pass 1 rate 3\n";
					DumpDeviceVector(rate3_device);

					*/

					forward_rate_extraction_gpu_main(  evolved_rates_device, 
						extractor1_cf,                          
						n_vectors,
						steps_to_test,
						number_rates, 
						rate1_cf_device                
						);

					forward_rate_extraction_gpu_main(  evolved_rates_device, 
						extractor2_cf,                          
						n_vectors,
						steps_to_test,
						number_rates, 
						rate2_cf_device                
						);

					forward_rate_extraction_gpu_main(  evolved_rates_device, 
						extractor3_cf,                          
						n_vectors,
						steps_to_test,
						number_rates, 
						rate3_cf_device                
						);

					cudaThreadSynchronize();

					/* for debugging only*/
					/*       
					for (int i=0; i < rate1_device.size(); ++i)
					{
					rate1_device[i] = i+batchCounter*100;
					rate2_device[i] = i*1000+batchCounter*111;
					rate3_device[i] = i*100000+batchCounter*201;


					}
					*/
					// end extra debugging code 

					rate1_host = rate1_device;
					rate2_host = rate2_device;
					rate3_host = rate3_device;
					std::copy(rate1_host.begin(), rate1_host.end(),rate1_vec.begin());
					std::copy(rate2_host.begin(), rate2_host.end(),rate2_vec.begin());
					std::copy(rate3_host.begin(), rate3_host.end(),rate3_vec.begin());

					rate1_cf_host = rate1_cf_device;
					rate2_cf_host = rate2_cf_device;
					rate3_cf_host = rate3_cf_device;
					std::copy(rate1_cf_host.begin(), rate1_cf_host.end(),rate1_cf_vec.begin());
					std::copy(rate2_cf_host.begin(), rate2_cf_host.end(),rate2_cf_vec.begin());
					std::copy(rate3_cf_host.begin(), rate3_cf_host.end(),rate3_cf_vec.begin());



					cudaThreadSynchronize();

					Timer h02;

					adjoinBasisVariablesCaller_main(useLogBasis,
						integer_Data_device, // data specific to the basis variables
						float_Data_device,
						evolved_rates_device,
						discounts_device,
						rate1_device,
						rate2_device,
						rate3_device,
						pathsPerBatch,
						totalPaths, 
						pathsPerBatch*batchCounter,
						numberOfRates, 
						exerciseIndices_vec.size(),
						exerciseIndices_device, // the indices of the exercise times amongst the evolution times
						exerciseIndicators_device, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
						final_basis_variables_device // output location
						);

					totalTimeForGPUExtraction += h02.timePassed();



					Timer h2;


					adjoinBasisVariablesCaller_gold(useLogBasis,
						integerData_vec, // int data specific to the basis variables
						floatData_vec, // float data specific to the basis variables
						forwards_cube,
						discountRatios_cube,
						rate1_matrix,
						rate2_matrix,
						rate3_matrix,
						pathsPerBatch,
						totalPaths, 
						pathsPerBatch*batchCounter,
						numberOfRates, 
						exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
						exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
						basisFunctionVariables_cube// output location
						);

					cudaThreadSynchronize();


					totalTimeForCPUExtraction += h2.timePassed();

					std::copy(discounts_host.begin(),discounts_host.end(),discounts_vec.begin());

					// spot measure numeraire



					Timer h3;

					spot_measure_numeraires_computation_gpu_offset_main(  discounts_device,
						spot_measure_values_device, //output
						pathsPerBatch,
						totalPaths,
						pathsPerBatch*batchCounter,
						numberOfRates
						);

					cudaThreadSynchronize();

					totalTimeForSpotMeasureGPU += h3.timePassed();

					Timer h4;

					spot_measure_numeraires_computation_offset_gold(  discounts_vec,
						pathsPerBatch,
						totalPaths,
						pathsPerBatch*batchCounter,
						numberOfRates,
						spot_measure_values_vec
						);

					cudaThreadSynchronize();

					totalTimeForSpotMeasureCPU +=h4.timePassed();

					// now get exercise values


					Timer h5;


					adjoinExerciseValues_Bermudan_swaption_main(exercise_values_device,  // output, one per exercise time per path
						float_Data_evalue_device, // any float auxiliary data for the exercise value object 
						integer_Data_evalue_device, //  any int auxiliary data for the exercise value object 
						exerciseIndices_device, // the indices of the exercise times amongst evolution times
						exerciseIndicators_device, // boolean indicators of exercise amongst evolution times 
						totalPaths, // typically totalPaths
						pathsPerBatch*batchCounter,
						pathsPerBatch, 
						numberOfRates, // assumed to be equal 
						rate1_device, 
						rate2_device, 
						rate3_device, 
						evolved_rates_device, 
						discounts_device);

					cudaThreadSynchronize();


					totalTimeForExerciseValuesGPU += h5.timePassed();

					Timer h6;

					adjoinExerciseValues_Bermudan_swaption_gold(float_Data_evalue_vec,
						integer_Data_evalue_vec, 
						numberOfRates,
						forwards_cube,
						discountRatios_cube,
						rate1_matrix,
						rate2_matrix,
						rate3_matrix,
						pathsPerBatch,
						pathsPerBatch*batchCounter, 
						exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
						exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
						exercise_values_matrix// output location
						);

					cudaThreadSynchronize();


					totalTimeForExerciseValuesCPU += h6.timePassed();




					Timer h7;

					cashFlowGeneratorCallerSwap_main(  genFlows1_device, 
						genFlows2_device, 
						aux_swap_data_device, 
						pathsPerBatch, 
						numberSteps,
						rate1_cf_device, 
						rate2_cf_device, 
						rate3_cf_device, 
						evolved_rates_device, 
						discounts_device);

					cudaThreadSynchronize();

					totalTimeForCashFlowGenerationGPU +=h7.timePassed();

					Timer h8;
					cashFlowGeneratorCallerSwap_gold(  genFlows1_cpu_vec, 
						genFlows2_cpu_vec, 
						aux_swap_data_vec, 
						pathsPerBatch, 
						numberSteps,
						rate1_cf_vec, 
						rate2_cf_vec, 
						rate3_cf_vec, 
						evolved_rates_gpu_vec, 
						discounts_vec);



					cudaThreadSynchronize();

					totalTimeForCashFlowGenerationCPU += h8.timePassed();

					cashFlow1Mismatches+= numberMismatches(genFlows1_cpu_matrix, 
						genFlows1_device, 
						misMatchTol,false);

					cashFlow2Mismatches+= numberMismatches(genFlows2_cpu_matrix, 
						genFlows2_device, 
						misMatchTol,false);


					spot_measure_numeraires_computation_gpu_offset_main(  discounts_device,
						spot_measure_values_small_device, //output
						pathsPerBatch,
						pathsPerBatch, // only want this batch
						0, // no offset
						numberOfRates
						);

					spot_measure_values_small_host = spot_measure_values_small_device;


					Timer h9;


					cashFlowDiscountingPartial_gold<float>(
						genTimeIndex_vec, // indivces of times at which flows are generated
						firstIndex_vec, // rate time index leq payment date 
						secondIndex_vec, // rate time index > payment date 
						thetas_vec, // interpolation fraction 
						deflation_locations_vec, // rate time index to discount to 
						discountRatios_cube,  // all the discount ratios 
						genFlows1_cpu_constmatrix, 
						spot_measure_values_small_constMatrix,
						pathsPerBatch, 
						numberSteps, 
						discounted_flows1_cpu_Matrix);

					cashFlowDiscountingPartial_gold<float>(
						genTimeIndex_vec, // indivces of times at which flows are generated
						firstIndex_vec, // rate time index leq payment date 
						secondIndex_vec, // rate time index > payment date 
						thetas_vec, // interpolation fraction 
						deflation_locations_vec, // rate time index to discount to 
						discountRatios_cube,  // all the discount ratios 
						genFlows2_cpu_constmatrix, 
						spot_measure_values_small_constMatrix,
						pathsPerBatch, 
						numberSteps, 
						discounted_flows2_cpu_Matrix);


					cudaThreadSynchronize();            
					totalTimeForCashFlowPartialDiscountingCPU +=h9.timePassed();

					Timer h10;


					cashFlowDiscounting_partial_gpu_main(genTimeIndex_device, // indivces of times at which flows are generated
						firstIndex_device, // rate time index leq payment date 
						secondIndex_device, // rate time index > payment date 
						thetas_device, // interpolation fraction 
						deflation_locations_device, 
						discounts_device, 
						genFlows1_device, 
						spot_measure_values_small_device,
						pathsPerBatch, 
						numberSteps, 
						discounted_flows1_device);

					cashFlowDiscounting_partial_gpu_main(genTimeIndex_device, // indivces of times at which flows are generated
						firstIndex_device, // rate time index leq payment date 
						secondIndex_device, // rate time index > payment date 
						thetas_device, // interpolation fraction 
						deflation_locations_device, 
						discounts_device, 
						genFlows2_device, 
						spot_measure_values_small_device,
						pathsPerBatch, 
						numberSteps, 
						discounted_flows2_device);

					cudaThreadSynchronize();

					totalTimeForCashFlowPartialDiscountingGPU += h10.timePassed();

					cashFlowDiscounting1_partial_mismatches+= numberMismatches(discounted_flows1_cpu_Matrix, 
						discounted_flows1_device, 
						misMatchTol,false);

					cashFlowDiscounting2_partial_mismatches+= numberMismatches(discounted_flows2_cpu_Matrix, 
						discounted_flows2_device, 
						misMatchTol,false);


					Timer h11;


					AggregateFlows_gold<float>(aggregatedFlows_cpu_matrix, // for output, aggregrates are added to existing data
						discounted_flows1_cpu_constMatrix,
						stepToExerciseIndices_vec, 
						pathsPerBatch,
						pathsPerBatch*batchCounter );

					AggregateFlows_gold<float>(aggregatedFlows_cpu_matrix, // for output, aggregrates are added to existing data
						discounted_flows2_cpu_constMatrix,
						stepToExerciseIndices_vec, 
						pathsPerBatch,
						pathsPerBatch*batchCounter );

					cudaThreadSynchronize();

					totalTimeForCashFlowAggregationCPU += h11.timePassed();

					Timer h12;





					AggregateFlows_main(
						aggregatedFlows_device,// output added to, not overwritten
						totalPaths,
						exerciseIndices_vec.size(), 
						discounted_flows1_device,
						pathsPerBatch, 
						pathsPerBatch*batchCounter,
						numberSteps, 
						stepToExerciseIndices_device                         );

					AggregateFlows_main(
						aggregatedFlows_device,// output added to, not overwritten
						totalPaths,
						exerciseIndices_vec.size(), 
						discounted_flows2_device,
						pathsPerBatch, 
						pathsPerBatch*batchCounter,
						numberSteps, 
						stepToExerciseIndices_device                         );


					cudaThreadSynchronize();

					totalTimeForCashFlowAggregationGPU += h12.timePassed();




				}//  end of   for (int batchCounter=0; batchCounter < numberOfBatches; ++batchCounter)

				std::cout << " finished loop...\n";


				std::cout << "Evolved rates \n";

				//   }// inner scope for destruction 

				std::cout << " exited inner scope\n";


				// now it's time to actually test!

				// first, we do the basis variables 
				thrust::host_vector<float> final_basis_variables_gpu_host(final_basis_variables_device);
				CubeConstFacade<float> final_basis_variables_gpu_cube(&final_basis_variables_gpu_host[0],exerciseIndices_vec.size(),basisVariableExample_gold::maxVariablesPerStep( ), totalPaths);

				int mismatches=0;

				std::cout << " entering test loop for extraction\n";

				std::vector<int> basisVariablesPerIndex(exerciseIndices_vec.size());
				for (int s=0; s < static_cast<int>(exerciseIndices_vec.size()); ++s)
					basisVariablesPerIndex[s]=  basisVariableExample_gold::actualNumberOfVariables(s, aliveIndex[s],number_rates);


				for (int p=0; p < totalPaths; ++p)
				{
					for (int s=0; s < static_cast<int>(exerciseIndices_vec.size()); ++s)
					{
						for (int v=0; v < basisVariablesPerIndex[s]; ++v)
						{                   
							if ( fabs(final_basis_variables_gpu_cube(s,v,p) - basisFunctionVariables_cube(s,v,p)) > tolerance)
							{
								++mismatches;
								std::cout <<" basis mismatch "  << p << ","<< s << ","<<v << ","<<final_basis_variables_gpu_cube(s,v,p) << ","<<basisFunctionVariables_cube(s,v,p) << "\n";

							}    


						}
					}
				}

				std::cout << " exiting test loop\n";

				if (mismatches>0)
				{
					std::cout << "Basis extraction failed\n";
					std::cout << "Number of mismatches " << mismatches << " out of "<<totalPaths*numberOfExerciseDates*basisVariableExample_gold::maxVariablesPerStep( ) << "\n";

				}
				else
				{
					std::cout << "Basis extraction succeeded.\n";
					--numberFails;

					std::cout << "CPU time " << totalTimeForCPUExtraction << "\n";
					std::cout << "GPU time " << totalTimeForGPUExtraction << "\n";
					std::cout << "CPU/GPU time " << totalTimeForCPUExtraction/totalTimeForGPUExtraction << " times speed up.\n";
				}

				// second we do the spot numeraires 

				thrust::host_vector<float> spots_host(spot_measure_values_device);       
				MatrixFacade<float> spots_gpu_matrix(&spots_host[0],numberSteps,totalPaths);
				MatrixFacade<float> spots_cpu_matrix(&spot_measure_values_vec[0],numberSteps,totalPaths);


				int spotMisMatches =0;


				for (int p=0; p < totalPaths; ++p)
				{
					for (int s=0; s < numberSteps; ++s)
					{
						//       std::cout << p << "," << s << "," << "," << spots_gpu_matrix(s,p) << "," << spots_cpu_matrix(s,p) << "\n";
						if (fabs((spots_gpu_matrix(s,p) - spots_cpu_matrix(s,p))/spots_cpu_matrix(s,p)) > misMatchTol)
							++spotMisMatches;
					}
				}

				if (spotMisMatches>0)
				{
					std::cout << "Spot measure failed\n";
					std::cout << "Number of mismatches " << spotMisMatches << " out of "<<totalPaths*numberSteps << "\n";

				}
				else
				{
					std::cout << "Spot measure succeeded.\n";
					--numberFails;

					std::cout << "CPU time " << totalTimeForSpotMeasureCPU << "\n";
					std::cout << "GPU time " << totalTimeForSpotMeasureGPU << "\n";
					std::cout << "CPU/GPU time " << totalTimeForSpotMeasureCPU/totalTimeForSpotMeasureGPU << " times speed up.\n";
				}

				// third, we test the exercise values 

				int exerciseErrs=  numberMismatches(exercise_values_matrix, 
					exercise_values_device, 
					misMatchTol,false);

				if (exerciseErrs>0)
				{
					std::cout << " exercise computation failed\n";
					std::cout << "Number of mismatches " << exerciseErrs << " out of "<<exercise_values_device.size() << "\n";

				}
				else
				{
					std::cout << "exercise computation succeeded.\n";
					--numberFails;

					std::cout << "CPU time " << totalTimeForExerciseValuesCPU << "\n";
					std::cout << "GPU time " << totalTimeForExerciseValuesGPU << "\n";
					std::cout << "CPU/GPU time " << totalTimeForExerciseValuesCPU/totalTimeForExerciseValuesGPU << " times speed up.\n";
				}

				// fourth cash-flow generation tests

				if (cashFlow1Mismatches +cashFlow2Mismatches >0)
				{
					std::cout << " cash flow computation failed\n";
					std::cout << "Number of mismatches " << cashFlow1Mismatches<< " and "<<cashFlow2Mismatches<< "\n";

				}
				else
				{
					std::cout << "cash flow computation succeeded.\n";
					--numberFails;

					std::cout << "CPU time " << totalTimeForCashFlowGenerationCPU << "\n";
					std::cout << "GPU time " << totalTimeForCashFlowGenerationGPU << "\n";
					std::cout << "CPU/GPU time " << totalTimeForCashFlowGenerationCPU/totalTimeForCashFlowGenerationGPU << " times speed up.\n";
				}

				// five cash-flow partial discounting tests

				if (cashFlowDiscounting1_partial_mismatches+cashFlowDiscounting2_partial_mismatches >0)
				{
					std::cout << " \n**\n**\ncash flow partial discounting failed\n";
					std::cout << "Number of mismatches " << cashFlowDiscounting1_partial_mismatches<< " and "<<cashFlowDiscounting2_partial_mismatches<< "\n\n\n";

				}
				else
				{
					std::cout << "cash flow partial discounting succeeded.\n";
					--numberFails;

					std::cout << "CPU time " << totalTimeForCashFlowPartialDiscountingCPU << "\n";
					std::cout << "GPU time " << totalTimeForCashFlowPartialDiscountingGPU << "\n";
					std::cout << "CPU/GPU time " << totalTimeForCashFlowPartialDiscountingCPU/totalTimeForCashFlowPartialDiscountingGPU << " times speed up.\n";
				}

				// six cash-flow aggregation tests


				int aggErrs=  numberMismatches(aggregatedFlows_cpu_matrix, 
					aggregatedFlows_device, 
					misMatchTol,false);

				if (aggErrs>0)
				{
					std::cout << " aggregation computation failed\n";
					std::cout << "Number of mismatches " << aggErrs << " out of "<<aggregatedFlows_device.size() << "\n";

				}
				else
				{
					std::cout << "aggregation computation succeeded.\n";
					--numberFails;
				}

				std::cout << "CPU time " << totalTimeForCashFlowAggregationCPU << "\n";
				std::cout << "GPU time " << totalTimeForCashFlowAggregationGPU << "\n";
				std::cout << "CPU/GPU time " << totalTimeForCashFlowAggregationCPU/totalTimeForCashFlowAggregationGPU << " times speed up.\n";

				// now test the least squares code...


				std::vector<int> basisVariablesEachStep(exerciseIndices_vec.size());

				std::vector<double> ls_coefficients;

				std::vector<float> some_numeraireValues_vec(exerciseIndices_vec.size()*totalPaths);

				MatrixFacade<float> some_numeraireValues_matrix(&some_numeraireValues_vec[0],exerciseIndices_vec.size(),totalPaths);

				spot_measure_numeraires_extraction_gold(   spot_measure_values_vec,
					totalPaths,
					numberSteps,
					exerciseIndices_vec.size(), 
					exerciseIndices_vec,
					some_numeraireValues_vec //output
					);


				thrust::device_vector<float> some_numeraireValues_device(exerciseIndices_vec.size()*totalPaths);

				spot_measure_numeraires_extraction_main(  spot_measure_values_device,
					some_numeraireValues_device, //output
					totalPaths,
					totalPaths,
					0,
					numberSteps,
					exerciseIndices_vec.size(), 
					exerciseIndices_device
					);
				// change types to Realv
				std::vector<Realv> basisFunctionVariables_realv_vec(final_basis_variables_vec.size());
				std::copy(final_basis_variables_vec.begin(),final_basis_variables_vec.end(),basisFunctionVariables_realv_vec.begin());

				CubeConstFacade<Realv> basisFunctionVariables_realv_cube(&basisFunctionVariables_realv_vec[0],
					basisFunctionVariables_cube.numberLayers(),
					basisFunctionVariables_cube.numberRows(),
					basisFunctionVariables_cube.numberColumns());

				std::vector<Realv> aggregatedFlows_realv_vec(aggregatedFlows_vec.size());
				std::copy(aggregatedFlows_vec.begin(),aggregatedFlows_vec.end(),aggregatedFlows_realv_vec.begin());

				MatrixFacade<Realv> aggregatedFlows_cpu_realv_matrix(&aggregatedFlows_realv_vec[0],
					aggregatedFlows_cpu_matrix.rows(),
					aggregatedFlows_cpu_matrix.columns());

				std::vector<Realv> exercise_values_realv_vec(exercise_values_vec.size());
				std::copy(exercise_values_vec.begin(),exercise_values_vec.end(),exercise_values_realv_vec.begin());

				MatrixFacade<Realv> exercise_values_realv_matrix(&exercise_values_realv_vec[0],
					exercise_values_matrix.rows(),
					exercise_values_matrix.columns());

				std::vector<Realv> exercise_values_zero_realv_vec(exercise_values_vec.size());
				std::fill(exercise_values_zero_realv_vec.begin(),exercise_values_zero_realv_vec.end(),0.0);

				MatrixFacade<Realv> exercise_values_zero_realv_matrix(&exercise_values_zero_realv_vec[0],
					exercise_values_matrix.rows(),
					exercise_values_matrix.columns());

				thrust::device_vector<float> exercise_values_zero_float_device(exercise_values_vec.size(),0.0f);


				std::vector<Realv> some_numeraireValues_realv_vec(some_numeraireValues_vec.size());
				std::copy(some_numeraireValues_vec.begin(),some_numeraireValues_vec.end(),some_numeraireValues_realv_vec.begin());

				MatrixFacade<Realv> some_numeraireValues_realv_matrix(&some_numeraireValues_realv_vec[0],
					some_numeraireValues_matrix.rows(),
					some_numeraireValues_matrix.columns());

				int numberBasisFunctions = 2*basisVariableExample_gold::maxVariablesPerStep( )+1;
				if (useCrossTerms)
					numberBasisFunctions = quadraticPolynomialCrossDevice::functionValues(basisVariableExample_gold::maxVariablesPerStep( ));


				std::vector<double> products_cube_vec;
				std::vector<double> targets_mat_vec;
				std::vector<double> productsM_cube_vec;
				std::vector<double> targetsM_mat_vec;

			
				std::vector<double> ls_coefficients_multi_vec(maxRegressionDepth*exerciseIndices_vec.size()*numberBasisFunctions);
				CubeFacade<double> regression_coefficients_cube(ls_coefficients_multi_vec,exerciseIndices_vec.size(),
																maxRegressionDepth,numberBasisFunctions);

				Matrix_gold<double> lowerCuts_mat(exerciseIndices_vec.size(),maxRegressionDepth,0.0);
				Matrix_gold<double> upperCuts_mat(exerciseIndices_vec.size(),maxRegressionDepth,0.0);


				double lsest = generateRegressionCoefficientsViaLSQuadratic_gold(exerciseIndices_vec.size(),
					products_cube_vec,
					targets_mat_vec,
					ls_coefficients, // output
					basisFunctionVariables_realv_cube,
					basisVariablesPerIndex,
					basisVariableExample_gold::maxVariablesPerStep( ),
					aggregatedFlows_cpu_realv_matrix, // deflated to current exercise time
					exercise_values_zero_realv_matrix, // deflated to current exercise time
					some_numeraireValues_realv_matrix,
					deflation_locations_vec,
					totalPaths,
					normalise,
					useCrossTerms);

				Cube_gold<Realv> means_cube_gold(exerciseIndices_vec.size(),numberOfExtraRegressions+1, basisVariableExample_gold::maxVariablesPerStep( ),0.0);
				Cube_gold<Realv> sds_cube_gold(exerciseIndices_vec.size(),numberOfExtraRegressions+1, basisVariableExample_gold::maxVariablesPerStep( ),0.0);

				double lsestM = generateRegressionCoefficientsViaLSMultiQuadratic_gold(exerciseIndices_vec.size(),
															  productsM_cube_vec,
															   targetsM_mat_vec,
															   regression_coefficients_cube,
															   lowerCuts_mat.Facade(),
															   upperCuts_mat.Facade(),
															   means_cube_gold.Facade(),
															   sds_cube_gold.Facade(),
															   normalise,
															   basisFunctionVariables_realv_cube,
															    basisVariablesPerIndex,
															   basisVariableExample_gold::maxVariablesPerStep( ),
															   aggregatedFlows_cpu_realv_matrix, // deflated to current exercise time
		                                              			exercise_values_zero_realv_matrix, // deflated to current exercise time
					                                           some_numeraireValues_realv_matrix,
					                                          deflation_locations_vec,
					                                          totalPaths,
															   numberOfExtraRegressions,
															  lowerPathCutoff,
															  regressionSelector,
															  useCrossTerms);

				std::cout << "LS est, " << lsest << " , LSEstM, " << lsestM << "\n";

				debugDumpCube(regression_coefficients_cube,"regression_coefficients_cube");

				std::vector<Realv> AndersenShifts_vec(exerciseIndices_vec.size());
				std::fill(AndersenShifts_vec.begin(),AndersenShifts_vec.end(),0.0);


				MatrixConstFacade<double> ls_coefficients_mat(ls_coefficients, exerciseIndices_vec.size(),ls_coefficients.size()/exerciseIndices_vec.size());


				double lsest2 =  SecondPassPriceLSAUsingAggregatedFlowsQuadratic_gold(exerciseIndices_vec.size(),
					ls_coefficients_mat, // output
					AndersenShifts_vec,
					basisFunctionVariables_realv_cube,
					basisVariablesPerIndex,
					basisVariableExample_gold::maxVariablesPerStep( ),
					aggregatedFlows_cpu_realv_matrix, // deflated to current exercise time
					exercise_values_zero_realv_matrix, // deflated to current exercise time
					some_numeraireValues_realv_matrix,
					deflation_locations_vec,
					totalPaths,
					useCrossTerms);

				double lsest2M;

				if (useCrossTerms)
				{
					MultiLSExerciseStrategy<quadraticPolynomialCrossGenerator> strategy(regression_coefficients_cube,
	                       lowerCuts_mat.Facade(),
							upperCuts_mat.Facade(),
							means_cube_gold.Facade(),
							sds_cube_gold.Facade(),
			                maxRegressionDepth,
						    AndersenShifts_vec,
							basisVariablesPerIndex,
							exerciseIndices_vec.size());
					

				   lsest2M= SecondPassPriceLSAMultiUsingAggregatedFlows_gold<MultiLSExerciseStrategy<quadraticPolynomialCrossGenerator> >(exerciseIndices_vec.size(),
												    basisFunctionVariables_realv_cube,
												    basisVariableExample_gold::maxVariablesPerStep( ),
												   aggregatedFlows_cpu_realv_matrix, 
												   exercise_values_zero_realv_matrix, 
												   some_numeraireValues_realv_matrix,
												   totalPaths,
												   deflation_locations_vec,
												   strategy
												   );


				}
				else
				{
						MultiLSExerciseStrategy<quadraticPolynomialGenerator> strategy(regression_coefficients_cube,
	                       lowerCuts_mat.Facade(),
							upperCuts_mat.Facade(),
							means_cube_gold.Facade(),
							sds_cube_gold.Facade(),
			                maxRegressionDepth,
						    AndersenShifts_vec,
							basisVariablesPerIndex,
							exerciseIndices_vec.size());
					

				   lsest2M= SecondPassPriceLSAMultiUsingAggregatedFlows_gold<MultiLSExerciseStrategy<quadraticPolynomialGenerator> >(exerciseIndices_vec.size(),
												    basisFunctionVariables_realv_cube,
												    basisVariableExample_gold::maxVariablesPerStep( ),
												   aggregatedFlows_cpu_realv_matrix, 
												   exercise_values_zero_realv_matrix, 
												   some_numeraireValues_realv_matrix,
												   totalPaths,
												   deflation_locations_vec,
												   strategy
												   );

				}


				if (fabs(lsest-lsest2 )> tolerance)
				{
					std::cout << " forwards and backwards LS estimates don't agree\n";
				}
				else 
					--numberFails;
					
				if (fabs(lsestM-lsest2M )> tolerance)
				{
					std::cout << " forwards and backwards LS estimates don't agree\n";
				}
				else 
					--numberFails;

				std::cout << "LS est2, " << lsest2 << " , LSEst2M, " << lsest2M << "\n";

				// also test generation of polynomials from variables

				int totalNumberBasisFunctions = useCrossTerms ? totalPaths*(3*basisVariableExample_gold::maxVariablesPerStep( )+1) : totalPaths*(2*basisVariableExample_gold::maxVariablesPerStep( )+1);

				std::vector<Realv> basisFunctions_vec(totalNumberBasisFunctions);
				int maxBasisVariables = basisVariableExample_gold::maxVariablesPerStep( );
				thrust::device_vector<float> basisFunctions_dev(totalNumberBasisFunctions);

				int errs=0;

				for (size_t i=0; i < exerciseIndices_vec.size(); ++i)
				{
					thrust::fill(basisFunctions_dev.begin(), basisFunctions_dev.end(),0.0f);
					std::fill(basisFunctions_vec.begin(), basisFunctions_vec.end(),0.0);

					expandBasisFunctions_quadratic_gold(
						totalPaths, 
						basisFunctionVariables_realv_vec, 
						i,
						exerciseIndices_vec.size(), 
						maxBasisVariables, 
						basisVariablesPerIndex[i],
						useCrossTerms,
						basisFunctions_vec
						);

					expandBasisFunctions_quadratic_main(
						totalPaths, 
						final_basis_variables_device, // input location for basis variables for all steps,
						i,
						basisVariableExample_gold::maxVariablesPerStep( ),
						basisVariablesPerIndex[i],
						useCrossTerms,
						basisFunctions_dev
						);


					int thiserrs = numberMismatches(basisFunctions_vec, basisFunctions_dev, misMatchTol, false);
					errs+= thiserrs;

					if (errs>0)
					{
						std::cout << " expand basis functions test failed for index "<< i <<  "\n";

//						debugDumpVector(basisFunctionVariables_realv_vec,"basisFunctionVariables_realv_vec");
//						debugDumpVector(basisFunctions_vec,"basisFunctions_vec");
//						debugDumpVector(final_basis_variables_device,"final_basis_variables_device");
//						debugDumpVector(basisFunctions_dev,"basisFunctions_dev");

					}


				}

				if (errs>0)
				{
					std::cout << " expand basis functions test failed with " << errs << " errors \n";



				}
				else
				{
					std::cout << " expand basis functions test passed\n";
					--numberFails;
				}

		
				thrust::host_vector<int> basisVariablesEachStep_host(basisVariablesPerIndex.begin(),basisVariablesPerIndex.end());
				thrust::device_vector<int> basisVariablesEachStep_device(basisVariablesEachStep_host);

				thrust::device_vector<int> basisVariableInt_data_device(numberExerciseDates,0);
				thrust::device_vector<float> basisVariableFloat_data_device(numberExerciseDates,0.0f);
				thrust::device_vector<float> coefficients_device(numberExerciseDates*numberBasisFunctions);
				std::vector<Realv> coefficients_vec(coefficients_device.size());

				int threads = 512;
				int blocks =3000;

				std::vector<float> products_gpu_cube_vec(numberExerciseDates*numberBasisFunctions*numberBasisFunctions);
				std::vector<float> targets_gpu_mat_vec(numberExerciseDates*numberBasisFunctions);

				bool useTestMode = true;
				bool testPassed=false;
				CubeConstFacade<double> products_gold_cube(products_cube_vec,numberExerciseDates,numberBasisFunctions,numberBasisFunctions);
				MatrixConstFacade<double>	targets_gold_mat(targets_mat_vec,numberExerciseDates,numberBasisFunctions);
				MatrixConstFacade<double> coefficients_gold(ls_coefficients,numberExerciseDates,numberBasisFunctions);
				double testTolerance = 1e-4;

				double LSest3= generateRegressionCoefficientsViaLSquadratic_gpu(numberExerciseDates,
					//		products_gpu_cube_vec,
					//          targets_gpu_mat_vec,
					basisVariableInt_data_device,
					basisVariableFloat_data_device,
					coefficients_device, // the LS coefficients are placed here 
					coefficients_vec, // the LS coefficients are also placed here 
					final_basis_variables_device, // cube of preextracted values for variables of basis funcions
					basisVariablesEachStep_device, //vector of the number of basis variables for each step
					basisVariablesPerIndex, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
					maxBasisVariables,
					aggregatedFlows_device, // deflated to current exercise time, matrix of times and paths 
					exercise_values_zero_float_device, // deflated to current exercise time, matrix of times and paths 
					some_numeraireValues_device, // numeraire vals for each, matrix of times and paths 
					deflation_locations_vec,
					totalPaths,
					threads,
					blocks,
					normalise,
					useCrossTerms,
					useTestMode,
					testPassed,
					products_gold_cube, // ignored if not in testmode
					targets_gold_mat,// ignored if not in testmode
					coefficients_gold,                                 // ignored if not in testmode
					testTolerance 
					);

				if (testPassed)
					--numberFails;

				bool testPasseddummy;

				if (fabs(lsest-LSest3 )> LStolerance)
				{
					std::cout << "  backwards  GPU and CPU LS estimates don't agree: ," << lsest << ", " << LSest3 <<"," <<lsest-LSest3 <<"\n";
					std::cout << " LS Coefficients for GPU and CPU are \n";
	/*				for (size_t i=0; i < coefficients_vec.size(); ++i)
					{
						std::cout << coefficients_vec[i] << "," << ls_coefficients[i] << "\n";
					}
					*/
					//			debugDumpCube(products_gpu_cube_vec, " Product Coefficients GPU are \n", numberExerciseDates, numberBasisFunctions,numberBasisFunctions );
					//			debugDumpCube(products_cube_vec, " Product Coefficients CPU are \n", numberExerciseDates, numberBasisFunctions,numberBasisFunctions );

					//			debugDumpMatrix(targets_gpu_mat_vec," Target Coefficients GPU are \n", numberExerciseDates, numberBasisFunctions);
					//			debugDumpMatrix(targets_mat_vec," Target Coefficients GPU are \n", numberExerciseDates, numberBasisFunctions);


				}
				else 
				{
					std::cout << " LS test in test mode passed : ," << lsest << ", " << LSest3 << " \n";
					--numberFails;
				}

			thrust::device_vector<float> lowercuts_device(numberExerciseDates*(numberOfExtraRegressions+1));
			thrust::device_vector<float> uppercuts_device(numberExerciseDates*(numberOfExtraRegressions+1));


			thrust::device_vector<float> coefficients_M_device(numberExerciseDates*(numberOfExtraRegressions+1)*maxBasisVariables);
			std::vector<double> coefficients_M_vec(coefficients_M_device.size());

		//	RegressionSelectorStandardDeviations regressionSelector(sdsForCutOff);

			RegressionSelectorFraction regressionSelector_gpu(lowerFrac, upperFrac,initialSDguess, multiplier);


			double LSest3M  
				= generateRegressionCoefficientsViaLSMultiquadratic_cula_main(numberExerciseDates,
				basisVariableInt_data_device,
				basisVariableFloat_data_device,
				coefficients_M_device, // the LS coefficients are placed here 
				coefficients_M_vec, // the LS coefficients are also placed here 
				lowercuts_device, // the lower cut points for the regressions 
				uppercuts_device, // the upper cut points for the regressions 									
				final_basis_variables_device, // cube of preextracted values for variables of basis funcions
				basisVariablesEachStep_device, //vector of the number of basis variables for each step
				basisVariablesPerIndex, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
				maxBasisVariables,
				aggregatedFlows_device, // deflated to current exercise time, matrix of times and paths 
				exercise_values_zero_float_device, // deflated to current exercise time, matrix of times and paths 
				some_numeraireValues_device, // numeraire vals for each, matrix of times and paths 
				deflation_locations_vec,
				totalPaths,
				useCrossTerms,
				numberOfExtraRegressions,
				regressionSelector_gpu,
				lowerPathCutoff);

			debugDumpCube(coefficients_M_vec,"coefficients_M_vec", numberExerciseDates,(numberOfExtraRegressions+1),numberBasisFunctions);


			std::cout << " backwards estimate using multi regression LS code " << LSest3M << "\n";


				double LSest4=-1.0;
				if (useCula)
					LSest4 = generateRegressionCoefficientsViaLSquadratic_cula_gpu(numberExerciseDates,
					basisVariableInt_data_device,
					basisVariableFloat_data_device,
					coefficients_device, // the LS coefficients are placed here 
					coefficients_vec, // the LS coefficients are also placed here 
					final_basis_variables_device, // cube of preextracted values for variables of basis funcions
					basisVariablesEachStep_device, //vector of the number of basis variables for each step
					basisVariablesPerIndex, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
					maxBasisVariables,
					aggregatedFlows_device, // deflated to current exercise time, matrix of times and paths 
					exercise_values_zero_float_device, // deflated to current exercise time, matrix of times and paths 
					some_numeraireValues_device, // numeraire vals for each, matrix of times and paths 
					deflation_locations_vec,
					totalPaths,
					threads,
					blocks,
					normalise,
					useCrossTerms
					);

				else
					LSest4 = generateRegressionCoefficientsViaLSquadratic_gpu(numberExerciseDates,
					basisVariableInt_data_device,
					basisVariableFloat_data_device,
					coefficients_device, // the LS coefficients are placed here 
					coefficients_vec, // the LS coefficients are also placed here 
					final_basis_variables_device, // cube of preextracted values for variables of basis funcions
					basisVariablesEachStep_device, //vector of the number of basis variables for each step
					basisVariablesPerIndex, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
					maxBasisVariables,
					aggregatedFlows_device, // deflated to current exercise time, matrix of times and paths 
					exercise_values_zero_float_device, // deflated to current exercise time, matrix of times and paths 
					some_numeraireValues_device, // numeraire vals for each, matrix of times and paths 
					deflation_locations_vec,
					totalPaths,
					threads,
					blocks,
					normalise,
					useCrossTerms,
					false, // testmode now off
					testPasseddummy,
					products_gold_cube, // ignored if not in testmode
					targets_gold_mat,// ignored if not in testmode
					coefficients_gold,                                 // ignored if not in testmode
					testTolerance 
					);



				if (fabs(lsest-LSest4 )> LStolerance)
				{
					std::cout << "  backwards  GPU and CPU LS non test mode estimates don't agree: ," << lsest << ", " << LSest4 <<"\n";
					std::cout << " LS Coefficients for GPU and CPU are \n";
			/*		for (size_t i=0; i < coefficients_vec.size(); ++i)
					{
						std::cout << coefficients_vec[i] << "," << ls_coefficients[i] << "\n";
					}
		*/			//				debugDumpCube(products_gpu_cube_vec, " Product Coefficients GPU are \n", numberExerciseDates, numberBasisFunctions,numberBasisFunctions );
					//				debugDumpCube(products_cube_vec, " Product Coefficients CPU are \n", numberExerciseDates, numberBasisFunctions,numberBasisFunctions );

					//				debugDumpMatrix(targets_gpu_mat_vec," Target Coefficients GPU are \n", numberExerciseDates, numberBasisFunctions);
					//				debugDumpMatrix(targets_mat_vec," Target Coefficients GPU are \n", numberExerciseDates, numberBasisFunctions);


				}
				else 
				{
					std::cout << " LS test in non test mode passed : ," << lsest << ", " << LSest4 << " \n";
					--numberFails;
				}

				// first passes are done
				// we still need to do the second passes!


				std::vector<float> batchValuesGold(numberOfSecondPassBatches);
				std::vector<float> batchValuesGPU_vec(numberOfSecondPassBatches);
				std::vector<float> batchValuesMGold(numberOfSecondPassBatches);
				std::vector<float> batchValuesMGPU_vec(numberOfSecondPassBatches);
			
				thrust::device_vector<float> batch_basis_variables_device(pathsPerSecondPassBatch*exerciseIndices_vec.size()*maxBasisVariables);

				thrust::device_vector<float> discountedFlows_dev(pathsPerSecondPassBatch*numberSteps);
				thrust::device_vector<float> summedDiscountedFlows1_dev(pathsPerSecondPassBatch);
				thrust::device_vector<float> summedDiscountedFlows2_dev(pathsPerSecondPassBatch);

				for (int pass=0; pass < 2; pass++)
				{
					std::vector<Realv> coefficientsToUse_vec( pass ==0 ? coefficients_vec : ls_coefficients );
					std::vector<Realv> coefficientsMultiToUse_vec( pass ==0 ? coefficients_M_vec : ls_coefficients_multi_vec );

					Matrix_gold<Realv> lowerCutsToUse_mat(pass ==0 ? MatrixCastDeviceVec<double,float>(lowercuts_device,numberExerciseDates,maxRegressionDepth) : lowerCuts_mat);
					Matrix_gold<Realv> upperCutsToUse_mat(pass ==0 ? MatrixCastDeviceVec<double,float>(uppercuts_device,numberExerciseDates,maxRegressionDepth) : upperCuts_mat);

					debugDumpMatrix(lowerCutsToUse_mat.ConstFacade(), "lowerCutsToUse_mat");
					debugDumpMatrix(upperCutsToUse_mat.ConstFacade(), "upperCutsToUse_mat");

					

					if (pass==0)
						std::cout << " \n\nDoing second pass using GPU generation for ls coefficients  \n";
					else
						std::cout << " \n\nDoing second pass using  using CPU generation for ls coefficients \n";

		//			debugDumpMatrix(coefficientsToUse_vec,"coefficientsToUse_vec\n",numberExerciseDates, numberBasisFunctions);

			//		for debugging 
			thrust::device_vector<float> estCVal_device(exerciseIndices_vec.size()*pathsPerSecondPassBatch);
                             
			thrust::device_vector<int>  exIndices_device(pathsPerSecondPassBatch);

					for (int k=0; k < numberOfSecondPassBatches; ++k)
					{
						int path_offset= numberOfBatches * pathsPerBatch*duplicate+k*pathsPerSecondPassBatch;

						bool doNormalInSobol=true;
						SobDevice( pathsPerSecondPassBatch, tot_dimensions,path_offset, device_input,doNormalInSobol);
						cudaThreadSynchronize();

						BrownianBridgeMultiDim::ordering allocator( BrownianBridgeMultiDim::triangular);


						MultiDBridge(pathsPerSecondPassBatch, 
							n_poweroftwo,
							factors,
							device_input, 
							dev_output,
							allocator,
							useTextures);

						correlated_drift_paths_device( dev_output,
							dev_correlated_rates, // correlated rate increments 
							A_device, // correlator 
							alive_device,
							drifts_device,
							factors*number_rates,
							factors, 
							number_rates,
							pathsPerSecondPassBatch,
							numberSteps);

						LMM_evolver_euler_main(  rates_device, 
							logRates_device, 
							taus_device,
							dev_correlated_rates,
							A_device,
							initial_drifts_device,
							displacements_device,
							aliveIndex,
							pathsPerSecondPassBatch,
							factors,
							steps_to_test, 
							number_rates, 
							e_buffer_device,
							evolved_rates_device, // for output
							evolved_log_rates_device  // for output 
							);






						evolved_rates_host =evolved_rates_device;

						evolved_rates_gpu_vec.resize(evolved_rates_host.size());
						std::copy(evolved_rates_host.begin(),evolved_rates_host.end(),evolved_rates_gpu_vec.begin());
						cudaThreadSynchronize();

						bool allStepsAtOnce =true;

						discount_ratios_computation_main( evolved_rates_device, 
							taus_device, 
							aliveIndex, 
							alive_device,
							pathsPerSecondPassBatch,
							steps_to_test, 
							number_rates, 
							discounts_device,
							allStepsAtOnce  
							);

						discounts_host=discounts_device;
						std::vector<float> discounts_vec(discounts_host.begin(),discounts_host.end());


						cudaThreadSynchronize();


						// set up rate vectors 

						coterminal_annuity_ratios_computation_gpu(  discounts_device, 
							taus_device, 
							aliveIndex, 
							n_vectors,
							steps_to_test, 
							number_rates, 
							annuities_device  // for output 
							);

						coterminal_swap_rates_computation_main_gpu(  discounts_device, 
							annuities_device, 
							aliveIndex, 
							n_vectors,
							steps_to_test, 
							number_rates, 
							swap_rates_device  // for output 
							);

						cudaThreadSynchronize();

						forward_rate_extraction_gpu_main(  evolved_rates_device, 
							extractor1,                          
							n_vectors,
							numberOfExerciseDates,
							number_rates, 
							rate1_device                
							);

						forward_rate_extraction_gpu_main(  swap_rates_device, 
							extractor2,                          
							n_vectors,
							numberOfExerciseDates,
							number_rates, 
							rate2_device                
							);

						forward_rate_extraction_gpu_main(  annuities_device, 
							extractor3,                          
							n_vectors,
							numberOfExerciseDates,
							number_rates, 
							rate3_device                
							);

						cudaThreadSynchronize();



						forward_rate_extraction_gpu_main(  evolved_rates_device, 
							extractor1_cf,                          
							n_vectors,
							numberSteps,
							number_rates, 
							rate1_cf_device                
							);

						forward_rate_extraction_gpu_main(  swap_rates_device, 
							extractor2_cf,                          
							n_vectors,
							numberSteps,
							number_rates, 
							rate2_cf_device                
							);

						forward_rate_extraction_gpu_main(  annuities_device, 
							extractor3_cf,                          
							n_vectors,
							numberSteps,
							number_rates, 
							rate3_cf_device                
							);

						cudaThreadSynchronize();


						rate1_cf_host = rate1_cf_device;
						rate2_cf_host = rate2_cf_device;
						rate3_cf_host = rate3_cf_device;
						std::copy(rate1_cf_host.begin(), rate1_cf_host.end(),rate1_cf_vec.begin());
						std::copy(rate2_cf_host.begin(), rate2_cf_host.end(),rate2_cf_vec.begin());
						std::copy(rate3_cf_host.begin(), rate3_cf_host.end(),rate3_cf_vec.begin());

						adjoinBasisVariablesCaller_main(useLogBasis,
							integer_Data_device, // data specific to the basis variables
							float_Data_device,
							evolved_rates_device,
							discounts_device,
							rate1_device,
							rate2_device,
							rate3_device,
							pathsPerSecondPassBatch,
							pathsPerSecondPassBatch, 
							0,
							numberOfRates, 
							exerciseIndices_vec.size(),
							exerciseIndices_device, // the indices of the exercise times amongst the evolution times
							exerciseIndicators_device, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
							batch_basis_variables_device // output location
							);

//						debugDumpVector(float_Data_device,"float_Data_device");
//						debugDumpVector(integer_Data_device,"integer_Data_device");


						//				debugDumpMatrix(rate1_device,"rate1_device",exerciseIndices_vec.size(),n_vectors);
						//					debugDumpMatrix(rate2_device,"rate2_device",exerciseIndices_vec.size(),n_vectors);
						//					debugDumpMatrix(rate3_device,"rate3_device",exerciseIndices_vec.size(),n_vectors);


						thrust::host_vector<float> batch_basis_variables_host(batch_basis_variables_device);
						std::vector<float> batch_basis_variables_vec(batch_basis_variables_host.begin(),batch_basis_variables_host.end());

						CubeConstFacade<float> batch_basis_variables_cube(batch_basis_variables_vec,
							exerciseIndices_vec.size(),
							maxBasisVariables,pathsPerSecondPassBatch);

				//							debugDumpCube(batch_basis_variables_vec,"batch_basis_variables_cube",
				//								exerciseIndices_vec.size(),
//												maxBasisVariables,pathsPerSecondPassBatch);
//

						thrust::device_vector<float> spot_measure_values_2_device(pathsPerSecondPassBatch*numberOfRates);

						spot_measure_numeraires_computation_gpu_offset_main(  discounts_device,
							spot_measure_values_2_device, //output
							pathsPerSecondPassBatch,
							pathsPerSecondPassBatch,
							0,
							numberOfRates
							);

						thrust::host_vector<float> spot_measure_values_2_host(spot_measure_values_2_device);
						std::vector<float> spot_measure_values_2_vec(spot_measure_values_2_host.begin(),spot_measure_values_2_host.end());



						std::vector<float> flow1_vec(pathsPerSecondPassBatch*numberOfRates);
						std::vector<float> flow2_vec(pathsPerSecondPassBatch*numberOfRates);

						earlyExerciseNullGold exerciseValue;

						CubeConstFacade<float> forwards2_cube(&evolved_rates_gpu_vec[0],numberSteps,numberOfRates,pathsPerSecondPassBatch);

						Swap_gold product( numberSteps, strike,payReceive,  accruals1, accruals2 );

						std::vector<bool> exerciseIndicatorsbool_vec(exerciseIndicators_vec.size());
						for (size_t i=0; i < exerciseIndicators_vec.size(); ++i)
							exerciseIndicatorsbool_vec[i] = exerciseIndicators_vec[i] != 0;


						if (useCrossTerms)
						{
							std::vector<quadraticPolynomialCrossGenerator> functionProducers;

							for (size_t i=0; i < exerciseIndices_vec.size(); ++i)
								functionProducers.push_back(quadraticPolynomialCrossGenerator(basisVariablesPerIndex[i]));

							LSAExerciseStrategy<quadraticPolynomialCrossGenerator> exStrategy(exerciseIndices_vec.size(),
								basisVariablesPerIndex,
								AndersenShifts_vec,
								coefficientsToUse_vec, 
								functionProducers
								);



							cashFlowGeneratorEE_gold(
								flow1_vec,
								flow2_vec, 
								product,
								exerciseValue,
								exStrategy,
								pathsPerSecondPassBatch, 
								numberSteps,
								exerciseIndicatorsbool_vec,
								rate1_cf_vec, 
								rate2_cf_vec, 
								rate3_cf_vec, 
								batch_basis_variables_cube,
								evolved_rates_gpu_vec, 
								discounts_vec);
						}
						else
						{
							std::vector<quadraticPolynomialGenerator> functionProducers;

							for (size_t i=0; i < exerciseIndices_vec.size(); ++i)
								functionProducers.push_back(quadraticPolynomialGenerator(basisVariablesPerIndex[i]));

							LSAExerciseStrategy<quadraticPolynomialGenerator> exStrategy(exerciseIndices_vec.size(),
								basisVariablesPerIndex,
								AndersenShifts_vec,
								coefficientsToUse_vec, 
								functionProducers
								);



							cashFlowGeneratorEE_gold(
								flow1_vec,
								flow2_vec, 
								product,
								exerciseValue,
								exStrategy,
								pathsPerSecondPassBatch, 
								numberSteps,
								exerciseIndicatorsbool_vec,
								rate1_cf_vec, 
								rate2_cf_vec, 
								rate3_cf_vec, 
								batch_basis_variables_cube,
								evolved_rates_gpu_vec, 
								discounts_vec);

						}


						std::vector<float> discountedFlows_vec(pathsPerSecondPassBatch*numberSteps);
						std::vector<float> summedDiscountedFlows1_vec(pathsPerSecondPassBatch);
						std::vector<float> summedDiscountedFlows2_vec(pathsPerSecondPassBatch);

						cashFlowDiscounting_gold(firstIndex_vec, 
							secondIndex_vec,
							thetas_vec, 
							discounts_vec, 
							flow1_vec, 
							spot_measure_values_2_vec,
							pathsPerSecondPassBatch, 
							numberSteps, 
							discountedFlows_vec, // output
							summedDiscountedFlows1_vec // output
							); 



						cashFlowDiscounting_gold(firstIndex_vec, 
							secondIndex_vec,
							thetas_vec, 
							discounts_vec, 
							flow2_vec, 
							spot_measure_values_2_vec,
							pathsPerSecondPassBatch, 
							numberSteps, 
							discountedFlows_vec, // output
							summedDiscountedFlows2_vec// output
							); 

						float batchSumGold = std::accumulate(summedDiscountedFlows1_vec.begin(),summedDiscountedFlows1_vec.end(),0.0f)
							+std::accumulate(summedDiscountedFlows2_vec.begin(),summedDiscountedFlows2_vec.end(),0.0f);

						batchValuesGold[k] = batchSumGold/ pathsPerSecondPassBatch;



						// multi gold version
						std::vector<float> flow1_multi_vec(pathsPerSecondPassBatch*numberOfRates);
						std::vector<float> flow2_multi_vec(pathsPerSecondPassBatch*numberOfRates);

						
						if (useCrossTerms)
						{
							std::vector<quadraticPolynomialCrossGenerator> functionProducers;

							for (size_t i=0; i < exerciseIndices_vec.size(); ++i)
								functionProducers.push_back(quadraticPolynomialCrossGenerator(basisVariablesPerIndex[i]));


					
							LSAMultiExerciseStrategy<quadraticPolynomialCrossGenerator> exMStrategy(static_cast<int>(exerciseIndices_vec.size()),
								basisVariablesPerIndex,
								AndersenShifts_vec,
								coefficientsMultiToUse_vec, 
								maxRegressionDepth,
		                        lowerCutsToUse_mat.ConstFacade(),
		                        upperCutsToUse_mat.ConstFacade(),
								means_cube_gold.ConstFacade(),
								sds_cube_gold.ConstFacade(),
								functionProducers
								);



							cashFlowGeneratorEE_gold(
								flow1_multi_vec,
								flow2_multi_vec, 
								product,
								exerciseValue,
								exMStrategy,
								pathsPerSecondPassBatch, 
								numberSteps,
								exerciseIndicatorsbool_vec,
								rate1_cf_vec, 
								rate2_cf_vec, 
								rate3_cf_vec, 
								batch_basis_variables_cube,
								evolved_rates_gpu_vec, 
								discounts_vec);
						}
						else
						{
							std::vector<quadraticPolynomialGenerator> functionProducers;

							for (size_t i=0; i < exerciseIndices_vec.size(); ++i)
								functionProducers.push_back(quadraticPolynomialGenerator(basisVariablesPerIndex[i]));

							LSAMultiExerciseStrategy<quadraticPolynomialGenerator> exMStrategy(static_cast<int>(exerciseIndices_vec.size()),
								basisVariablesPerIndex,
								AndersenShifts_vec,
								coefficientsMultiToUse_vec, 
								maxRegressionDepth,
		                        lowerCutsToUse_mat.ConstFacade(),
		                        upperCutsToUse_mat.ConstFacade(),
								means_cube_gold.ConstFacade(),
								sds_cube_gold.ConstFacade(),
								functionProducers
								);

							cashFlowGeneratorEE_gold(
								flow1_multi_vec,
								flow2_multi_vec, 
								product,
								exerciseValue,
								exMStrategy,
								pathsPerSecondPassBatch, 
								numberSteps,
								exerciseIndicatorsbool_vec,
								rate1_cf_vec, 
								rate2_cf_vec, 
								rate3_cf_vec, 
								batch_basis_variables_cube,
								evolved_rates_gpu_vec, 
								discounts_vec);

						}

						std::vector<float> discountedFlows_M_vec(pathsPerSecondPassBatch*numberSteps);
						std::vector<float> summedDiscountedFlows1_M_vec(pathsPerSecondPassBatch);
						std::vector<float> summedDiscountedFlows2_M_vec(pathsPerSecondPassBatch);

						cashFlowDiscounting_gold(firstIndex_vec, 
							secondIndex_vec,
							thetas_vec, 
							discounts_vec, 
							flow1_multi_vec, 
							spot_measure_values_2_vec,
							pathsPerSecondPassBatch, 
							numberSteps, 
							discountedFlows_M_vec, // output
							summedDiscountedFlows1_M_vec // output
							); 



						cashFlowDiscounting_gold(firstIndex_vec, 
							secondIndex_vec,
							thetas_vec, 
							discounts_vec, 
							flow2_multi_vec, 
							spot_measure_values_2_vec,
							pathsPerSecondPassBatch, 
							numberSteps, 
							discountedFlows_M_vec, // output
							summedDiscountedFlows2_M_vec// output
							); 

						
					/*	if (pass ==0)
						{
							debugDumpVector(summedDiscountedFlows1_M_vec,"summedDiscountedFlows1_M_vec");
							debugDumpVector(summedDiscountedFlows2_M_vec,"summedDiscountedFlows2_M_vec");
						}
*/

						float batchSumMGold = std::accumulate(summedDiscountedFlows1_M_vec.begin(),summedDiscountedFlows1_M_vec.end(),0.0f)
							+std::accumulate(summedDiscountedFlows2_M_vec.begin(),summedDiscountedFlows2_M_vec.end(),0.0f);

						batchValuesMGold[k] = batchSumMGold/ pathsPerSecondPassBatch;



						


						// now for the GPU version

						thrust::host_vector<bool> exerciseIndicatorsbool_host(exerciseIndicatorsbool_vec.begin(),exerciseIndicatorsbool_vec.end());
						thrust::device_vector<bool> exerciseIndicatorsbool_dev(exerciseIndicatorsbool_host);

						thrust::device_vector<float> exerciseValueDataFloatdummy_device(1,0.0f);
						thrust::device_vector<int> exerciseValueDataIntdummy_device(1,0);

		//				thrust::device_vector<float> estCVal_device(pathsPerSecondPassBatch*numberExerciseDates);
		//				thrust::device_vector<int> exIndices_device(pathsPerSecondPassBatch);


						int floatDataSize;
						int intDataSize;

						if (useCrossTerms)

						{

							LSAExerciseStrategyQuadraticCross_gpu::outputDataVectorSize(floatDataSize,
								intDataSize, 
								exerciseIndices_vec.size(),
								basisVariablesPerIndex);


							thrust::device_vector<float>  exerciseStrategyDataFloat_device( floatDataSize);                                 
							thrust::device_vector<int>  exerciseStrategyDataInt_device( intDataSize);     

							LSAExerciseStrategyQuadraticCross_gpu::outputDataVectors(exerciseStrategyDataFloat_device.begin(), 
								exerciseStrategyDataInt_device.begin(),
								static_cast<int>(exerciseIndices_vec.size()),
								basisVariablesPerIndex ,
								AndersenShifts_vec,
								coefficientsToUse_vec);



							cashFlowGeneratorEE_main<Swap ,earlyExerciseNull ,LSAExerciseStrategyQuadraticCross_gpu >(
								genFlows1_device,
								genFlows2_device, 
								aux_swap_data_device,
								exerciseValueDataFloatdummy_device,
								exerciseValueDataIntdummy_device,
								exerciseStrategyDataFloat_device,
								floatDataSize,
								exerciseStrategyDataInt_device,
								intDataSize,
								pathsPerSecondPassBatch, 
								numberSteps,
								static_cast<int>(exerciseIndices_vec.size()),
								exerciseIndicatorsbool_dev,
								rate1_cf_device, 
								rate2_cf_device, 
								rate3_cf_device, 
								batch_basis_variables_device,
								maxBasisVariables,
								evolved_rates_device, 
								discounts_device
								,
								estCVal_device,
								exIndices_device
								);
						}
						else
						{

							LSAExerciseStrategyQuadratic_gpu::outputDataVectorSize(floatDataSize,
								intDataSize, 
								exerciseIndices_vec.size(),
								basisVariablesPerIndex);


							thrust::device_vector<float>  exerciseStrategyDataFloat_device( floatDataSize);                                 
							thrust::device_vector<int>  exerciseStrategyDataInt_device( intDataSize);     

							LSAExerciseStrategyQuadratic_gpu::outputDataVectors(exerciseStrategyDataFloat_device.begin(), 
								exerciseStrategyDataInt_device.begin(),
								static_cast<int>(exerciseIndices_vec.size()),
								basisVariablesPerIndex ,
								AndersenShifts_vec,
								coefficientsToUse_vec);



							cashFlowGeneratorEE_main<Swap ,earlyExerciseNull ,LSAExerciseStrategyQuadratic_gpu >(
								genFlows1_device,
								genFlows2_device, 
								aux_swap_data_device,
								exerciseValueDataFloatdummy_device,
								exerciseValueDataIntdummy_device,
								exerciseStrategyDataFloat_device,
								floatDataSize,
								exerciseStrategyDataInt_device,
								intDataSize,
								pathsPerSecondPassBatch, 
								numberSteps,
								static_cast<int>(exerciseIndices_vec.size()),
								exerciseIndicatorsbool_dev,
								rate1_cf_device, 
								rate2_cf_device, 
								rate3_cf_device, 
								batch_basis_variables_device,
								maxBasisVariables,
								evolved_rates_device, 
								discounts_device
								,
								estCVal_device,
								exIndices_device
							);
						}

						// compare with CPU version of genFlows


						std::vector<float> genFlows1_gpu_vec(stlVecFromDevVec<float>(genFlows1_device));

						int genFlow1Errs=  numberMismatches<float,float>(flow1_vec, genFlows1_gpu_vec, 
							misMatchTol,false);

						std::cout << "number of genFlows1 mismatches, for this batch " << genFlow1Errs << "\n";

		//				std::vector<float> estVal_gpu_vec(stlVecFromDevVec(estCVal_device));
		//	            std::vector<int> exIndices_gpu_vec(stlVecFromDevVec(exIndices_device));

		//				debugDumpMatrix(estVal_gpu_vec,"estVal_gpu_vec",numberExerciseDates,pathsPerSecondPassBatch);
		//				debugDumpVector(exIndices_gpu_vec,"exIndices_gpu_vec");




						cashFlowDiscounting_gpu_main(firstIndex_device, 
							secondIndex_device,
							thetas_device, 
							discounts_device, 
							genFlows1_device, 
							spot_measure_values_2_device,
							pathsPerSecondPassBatch, 
							numberSteps, 
							useTexturesForDiscounting,
							useSharedForDiscounting,
							discountedFlows_dev, // output
							summedDiscountedFlows1_dev); // output

						cashFlowDiscounting_gpu_main(firstIndex_device, 
							secondIndex_device,
							thetas_device, 
							discounts_device, 
							genFlows2_device, 
							spot_measure_values_2_device,
							pathsPerSecondPassBatch, 
							numberSteps, 
							useTexturesForDiscounting,
							useSharedForDiscounting,
							discountedFlows_dev, // output
							summedDiscountedFlows2_dev); // output

						float batchValueGPU = thrust::reduce(summedDiscountedFlows1_dev.begin(),summedDiscountedFlows1_dev.end());
						batchValueGPU += thrust::reduce(summedDiscountedFlows2_dev.begin(),summedDiscountedFlows2_dev.end());
						batchValueGPU /= pathsPerSecondPassBatch;

						batchValuesGPU_vec[k] = batchValueGPU;

					}

					float valueGold = std::accumulate(batchValuesGold.begin(),batchValuesGold.end(),0.0f)/numberOfSecondPassBatches;
					float valueMGold = std::accumulate(batchValuesMGold.begin(),batchValuesMGold.end(),0.0f)/numberOfSecondPassBatches;
				
					float valueGPU = std::accumulate(batchValuesGPU_vec.begin(),batchValuesGPU_vec.end(),0.0f)/numberOfSecondPassBatches;

					std::cout << " first pass value, " << LSest3 << "\n";
					std::cout << " second pass  gold value, " << valueGold << "\n";
					std::cout << " second pass  gold multi value, " << valueMGold << "\n";
			
					std::cout << " second pass  gpu value, " << valueGPU<< "\n";


					std::cout << " second pass cpu values :\n";

					for (size_t i=0; i < batchValuesGold.size(); ++i)
						std::cout << batchValuesGold[i] << ",";
					std::cout << "\n";


					std::cout << " second pass gpu values :\n";

					for (size_t i=0; i < batchValuesGPU_vec.size(); ++i)
						std::cout << batchValuesGPU_vec[i] << ",";
					std::cout << "\n";

				}
			}

		} // outer scope for destruction
	}
	cudaThreadExit();

	return numberFails;
}
