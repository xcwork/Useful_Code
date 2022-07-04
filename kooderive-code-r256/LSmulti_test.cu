//
//
//       LSmulti_Test.cu
//
//
// (c) Mark Joshi 2011,2012,2013,2014
// This code is released under the GNU public licence version 3

// routine to test the multi LS Code

#include <LSmulti_test.h>
#include <gold/MultiLS_regression.h>
#include <gold\Regression_Selector_concrete_gold.h>
#include <LS_Basis_main.h>
#include <gold/LS_Basis_gold.h>
#include <cutil_inline.h>
#include <cutil.h>
#include <gold/math/cube_gold.h>
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
#include <LMM_evolver_full.h>
#include <numeric>
#include <RegressionSelectorConcrete.h>
#include <smallFunctions.h>
#include <LinearSolverConcrete_main.h>
namespace
{
	double tolerance = 1E-6;
	double misMatchTol = 1E-4;
	double lsfirstPassTol = 1E-3;
}


int Test_Multi_LS_etc(bool verbose,DeviceChooser& chooser)
{
	int fails=0; 
		
	for (int useCrossTerms=0; useCrossTerms <2; useCrossTerms++)
		for (int useLogBasis =0; useLogBasis <2; useLogBasis++)
			for (int globalNormalise=0; globalNormalise <2; ++globalNormalise)		
				fails +=Test_LS__Multi_Code( verbose, useLogBasis>0,  useCrossTerms>0,globalNormalise>0,chooser);
				
			

	return fails;

}

int Test_LS__Multi_Code(bool verbose, bool useLogBasis,bool useCrossTerms,  bool globallyNormalise,DeviceChooser& chooser)
{
	bool useShared = false;
	bool useSharedMem = true;
	bool doDiscounts =true;
	bool newBridge =true;
	bool fermiArch = false;


	if (verbose)
	{
		std::cout << "\n\nTesting Test_LS__Multi_Code  routine ";           
	}
	int numberFails =10;
	std::vector<float> xa(4);
	xa[0] = 0.01f;
	xa[1] = 0.005f;
	xa[2] = 0.0f;
	xa[3] = 0.025f;

	int numberNonCallDates =2;
	int choice =2;

	std::vector<int> rates(4);
	rates[0] = 10+numberNonCallDates;
	rates[1] = 18+numberNonCallDates;
	rates[2] = 38+numberNonCallDates;
	rates[3] = 1+numberNonCallDates;

	int numberOfRates = rates[choice];

	int n_poweroftwo =1;
	{
		int power =2;
		while (power < numberOfRates)
		{
			++n_poweroftwo;
			power*=2;
		}
	}

	float x =xa[choice];
	float firstForward=0.008f+x;
	float forwardIncrement=0.002f;
	firstForward += (3-numberNonCallDates)*forwardIncrement;
	float displacement=0.015f;
	float strike =0.04f;

	float beta=2*0.0669f;				  
	float L=0.0f;
	double a= 0.05;
	double b = 0.09;
	double c= 0.44;
	double d=0.2;
	bool useFlatVols = false;
	double rateLength = 0.5;

	double firstRateTime =1.5-rateLength*numberNonCallDates;
	if (firstRateTime < 0.5)
		firstRateTime =0.5;


	int numberSteps = numberOfRates;
	int number_rates = numberOfRates;
	// dummy calibration data 
	std::vector<float> logRates_vec(number_rates);
	std::vector<float> rates_vec(number_rates);
	std::vector<float> taus_vec(number_rates);
	std::vector<float> displacements_vec(number_rates);
	std::vector<float> rateTimes_vec(number_rates+1);
	std::vector<float> evolutionTimes_vec(number_rates);

	std::vector<double> rateTimes_d_vec(number_rates+1);
	std::vector<double> evolutionTimes_d_vec(number_rates);

	rateTimes_vec[0] =static_cast<float>(firstRateTime);
	rateTimes_d_vec[0] =firstRateTime;

	for (int i=0; i < number_rates; ++i)
	{
		rates_vec[i] = firstForward+i*forwardIncrement;
		displacements_vec[i] =displacement;
		logRates_vec[i] = logf(rates_vec[i]+displacements_vec[i] );
		taus_vec[i] = static_cast<float>(rateLength);
		rateTimes_d_vec[i+1] = firstRateTime + rateLength*(i+1);
		rateTimes_vec[i+1]= static_cast<float>(firstRateTime + rateLength*(i+1));
		evolutionTimes_d_vec[i] = rateTimes_d_vec[i]; 
		evolutionTimes_vec[i] = rateTimes_vec[i];

	}   

	float maxDisplacement = *std::max_element(displacements_vec.begin(),displacements_vec.end());

	double beta_d = 2*0.0668;
	double L_d = 0.0;
	float vol = 0.11f;

	std::vector<float> vols(number_rates,vol);
	std::vector<double> vols_d(number_rates,vol);


	std::vector<double> Ks_vec(number_rates,1.0);

	int factors = min(number_rates,5);

	Cube_gold<double> pseudosDouble(useFlatVols ? FlatVolPseudoRootsOfCovariances(rateTimes_d_vec,
		evolutionTimes_d_vec,
		vols_d,
		factors,
		L_d,
		beta_d
		) : 
	ABCDLBetaPseudoRoots(a,b,c,d,
		evolutionTimes_d_vec,
		evolutionTimes_d_vec,
		Ks_vec,
		factors,
		L,
		beta
		));

	//		debugDumpCube<double>(pseudosDouble.ConstFacade(),"pseudos");

	Cube_gold<float> pseudos(CubeTypeConvert<float,double>(pseudosDouble));


	size_t pathsPerBatch = 32767;
	int lowerPathCutoff =1000;

	size_t mb=2;
	size_t maxb=3;

#ifdef _DEBUG
	pathsPerBatch = 8191;
	lowerPathCutoff = 64;
	mb = 1;
	maxb=2;
#endif
    		cudaSetDevice(chooser.WhichDevice());

    	 size_t outN = 5*pathsPerBatch*numberSteps*number_rates+5*maxb*pathsPerBatch*numberSteps;
            
         size_t estimatedFloatsRequired = outN;
   
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
               outN= 10*pathsPerBatch*numberSteps*number_rates+10*maxb*pathsPerBatch*numberSteps;
            
  //             change =true;
               estimatedFloatsRequired = outN;
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
     
           }
       }

	int numberOfExtraRegressions =2;
	int maxRegressionDepth = numberOfExtraRegressions+1;


	int duplicate=1; // if zero use same paths as for first pass, if 1 start at the next paths



	double lowerFrac = 0.25;
	double upperFrac=0.26;
	double initialSDguess = 2.5;
	double multiplier =0.8;
	RegressionSelector_gold_Fraction regressionSelector(lowerFrac, upperFrac, initialSDguess,  multiplier);




	for(	int numberOfBatches = mb; numberOfBatches < maxb;++numberOfBatches)
	{
		numberFails =10;	

		std::cout << "using " << numberOfBatches << " of " << pathsPerBatch << " for first pass\n";

		int numberOfSecondPassBatches = numberOfBatches;
		int pathsPerSecondPassBatch = pathsPerBatch; 

		std:: cout << "pathsPerBatch,"<< pathsPerBatch <<",numberOfBatches ," << numberOfBatches 
			<<",numberOfSecondPassBatches," << numberOfSecondPassBatches<<",useCrossTerms," 
			<< useCrossTerms<<",regressionDepth," << maxRegressionDepth << ",sdsForCutOff,"
			<< initialSDguess
			<<",minPathsForRegression," <<				    lowerPathCutoff
			<<",lowerFrac," <<					lowerFrac
			<<",upperFrac," <<				    upperFrac
			<<",multiplier," <<					multiplier
			<<",globallyNormalise," <<				    globallyNormalise
			<<",duplicate," <<				    duplicate << "\n";


		int totalPaths = pathsPerBatch*numberOfBatches;

		int n_vectors =  pathsPerBatch;


		int tot_dimensions = numberSteps*factors;
		int N= n_vectors*tot_dimensions; 

		int numberOfRates = number_rates;
		int numberOfSteps = numberSteps;

		int basisVariablesPerStep = basisVariableExample_gold<float>::maxVariablesPerStep();


		bool useTextures = true;

		std::cout << " uselog basis," << useLogBasis  << ", useCrossTerms , " 
			<< useCrossTerms  <<"\n";

		double totalTimeForGPUExtraction=0.0;
		//		double totalTimeForCPUExtraction=0.0;

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

		double totalTimeForCPUExtraction=0.0;


		int  cashFlow1Mismatches=0; 
		int  cashFlow2Mismatches=0;

		int cashFlowDiscounting1_partial_mismatches=0;
		int cashFlowDiscounting2_partial_mismatches=0;



		cudaThreadSynchronize();

		// set up all device vectors first
		// let's put everything inside a little scope so it all destroys earlier
		{
			std::vector<int> exerciseIndices_vec; // the indices of the exercise times amongst the evolution times
			std::vector<int> exerciseIndicators_vec(numberSteps,0);
			for (int i=numberNonCallDates; i < numberSteps;++i) // 
			{
				exerciseIndicators_vec[i] =1;
			}

			//   exerciseIndicators_vec[1] =0;


			GenerateIndices(exerciseIndicators_vec, exerciseIndices_vec);

			int numberOfExerciseDates = exerciseIndices_vec.size();

			int totalNumberOfBasisVariables = basisVariablesPerStep*numberOfExerciseDates*totalPaths;

			Cube_gold<float> basisFunctionVariables_gold_cube(numberOfExerciseDates,basisVariablesPerStep,totalPaths,0.0f);


			// across batch data
			thrust::device_vector<float> final_basis_variables_device(totalNumberOfBasisVariables);

			thrust::device_vector<float> spot_measure_values_device(numberOfSteps*totalPaths);
			std::vector<float> spot_measure_values_vec(numberOfSteps*totalPaths);
			MatrixFacade<float>  spot_measure_values_matrix(&spot_measure_values_vec[0],numberOfSteps,totalPaths);




			thrust::device_vector<int> alive_device;   
			int steps_to_test  = numberSteps; // must be less than or equal to numberSteps


			thrust::host_vector<int> exerciseIndices_host(exerciseIndices_vec.begin(),exerciseIndices_vec.end());
			thrust::host_vector<int> exerciseIndicators_host(exerciseIndicators_vec.begin(),exerciseIndicators_vec.end());


			thrust::device_vector<int> exerciseIndices_device(exerciseIndices_host);
			thrust::device_vector<int> exerciseIndicators_device(exerciseIndicators_host);

			thrust::host_vector<bool> exerciseIndicatorsBool_host(exerciseIndicators_host.size());

			for (size_t i=0; i < exerciseIndicatorsBool_host.size(); ++i)
				exerciseIndicatorsBool_host[i] = exerciseIndicators_host[i]  != 0;

			thrust::device_vector<bool> exerciseIndicatorsBool_device(exerciseIndicatorsBool_host);


			//	int numberExerciseDates =  exerciseIndices_vec.size();
			int totalNumberOfExerciseValues =numberOfExerciseDates*totalPaths;
			thrust::device_vector<float> exercise_values_device(totalNumberOfExerciseValues);

			//     for (int i=0; i < totalNumberOfExerciseValues; ++i)
			//      exercise_values_device[i] = i;

			std::vector<float> exercise_values_vec(totalNumberOfExerciseValues);
			MatrixFacade<float>  exercise_values_matrix(&exercise_values_vec[0],exerciseIndices_vec.size(),totalPaths);


			int outN = n_vectors*numberSteps*number_rates;

			std::vector<int> deflation_locations_vec(exerciseIndices_vec);
			std::vector<int> aliveIndex(numberSteps);
			for (int i=0; i < numberSteps; ++i)
			{
				aliveIndex[i] = i;

			}

			thrust::device_vector<int> deflation_locations_device(deflation_locations_vec.begin(),deflation_locations_vec.end());

			thrust::device_vector<float> evolved_rates_device(outN);


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

			thrust::device_vector<float> aggregatedFlows_device(totalPaths*exerciseIndices_vec.size(),0.0f);

			std::vector<float> aggregatedFlows_vec(totalPaths*exerciseIndices_vec.size(),0.0f);
			MatrixFacade<float> aggregatedFlows_cpu_matrix(&aggregatedFlows_vec[0],exerciseIndices_vec.size(),totalPaths);


			// create new scope so that everything inside dies at end of it
			{

				thrust::host_vector<int> alive_host(aliveIndex.begin(),aliveIndex.end());
				alive_device=alive_host;

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

					spotDriftComputer<float> driftComp(taus_vec,  factors,displacements_vec);
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

				int firstPositiveStep = findExerciseTimeIndicesFromPaymentTimeIndices(genTimeIndex_vec, 
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
				std::vector<float> aux_swap_2_data_vec(2*number_rates+2);

				float payReceive = 1.0f; // i.e. receive fixed
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
				aux_swap_2_data_vec = aux_swap_data_vec;

				for (int j=0; j < numberNonCallDates; ++j)
				{
					aux_swap_2_data_vec[j+2] =0.0;
					aux_swap_2_data_vec[number_rates+j+2] =  0.0;
				}

				thrust::device_vector<float> aux_swap_data_2_device(aux_swap_2_data_vec.begin(),aux_swap_2_data_vec.end());



				thrust::device_vector<float> spot_measure_values_small_device(n_vectors*steps_to_test);
				thrust::host_vector<float> spot_measure_values_small_host(n_vectors*steps_to_test);
				MatrixFacade<float> spot_measure_values_small_Matrix(&spot_measure_values_small_host[0],steps_to_test,n_vectors);
				MatrixConstFacade<float> spot_measure_values_small_constMatrix(&spot_measure_values_small_host[0],steps_to_test,n_vectors);


				std::vector<int> fromGenIndicesToExerciseDeflations_vec(genTimeIndex_vec.size());
				for (size_t i=0; i < genTimeIndex_vec.size();++i)
				{

					int j=stepToExerciseIndices_vec[i];

					fromGenIndicesToExerciseDeflations_vec[i] = j>=0 ? deflation_locations_vec[j] :0 ;
				}
				thrust::host_vector<int> fromGenIndicesToExerciseDeflations_host(fromGenIndicesToExerciseDeflations_vec.begin(),
					fromGenIndicesToExerciseDeflations_vec.end());

				thrust::device_vector<int> fromGenIndicesToExerciseDeflations_device(fromGenIndicesToExerciseDeflations_host);

				// every is now set up so can do  loops 

				thrust::device_vector<unsigned int> scrambler_device(tot_dimensions,0);
				thrust::device_vector<unsigned int> SobolInts_buffer_device(N);

				thrust::device_vector<float> e_pred_buffer_device(factors*n_vectors);

				cudaThreadSynchronize();

				for (int batchCounter=0; batchCounter < numberOfBatches; ++batchCounter)
				{
					int path_offset= batchCounter * pathsPerBatch;

					LMMEvolutionRoutineRawSingleKernel(pathsPerBatch, 
						path_offset, 
						number_rates, 
						factors, 
						number_rates, 
						n_poweroftwo,
						scrambler_device, 
						A_device,
						drifts_device, 
						displacements_device,
						rates_device, 
						logRates_device, 
						taus_device, 
						initial_drifts_device, 
						aliveIndex, 
						alive_device, 
						// buffers
						SobolInts_buffer_device, 
						device_input, 
						dev_output, 
						thrust::raw_pointer_cast(&dev_correlated_rates[0]), 
						thrust::raw_pointer_cast(&e_buffer_device[0]),
						thrust::raw_pointer_cast(&e_pred_buffer_device[0]),
						thrust::raw_pointer_cast(&evolved_rates_device[0]),
						thrust::raw_pointer_cast(&evolved_log_rates_device[0]),
						thrust::raw_pointer_cast(&discounts_device[0]),
						useSharedMem,
						doDiscounts,
						newBridge,
						fermiArch,
						0 // threads
						);


					cudaThreadSynchronize();



					evolved_rates_host =evolved_rates_device;



					evolved_rates_gpu_vec.resize(evolved_rates_host.size());
					std::copy(evolved_rates_host.begin(),evolved_rates_host.end(),evolved_rates_gpu_vec.begin());
					cudaThreadSynchronize();

					//			debugDumpCube(evolved_rates_gpu_vec,"evolved rates", steps_to_test,number_rates,n_vectors);

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



					forward_rate_extraction_steps_gpu_main(  evolved_rates_device, 
						extractor1,
						exerciseIndices_vec,
						n_vectors,
						number_rates, 
						rate1_device                
						);

					forward_rate_extraction_steps_gpu_main(//evolved_rates_device, 
						swap_rates_device, 
						extractor2,   
						exerciseIndices_vec,
						n_vectors,
						number_rates, 
						rate2_device                
						);

					forward_rate_extraction_steps_gpu_main( // evolved_rates_device,
						annuities_device, 
						extractor3,       
						exerciseIndices_vec,
						n_vectors,
						number_rates, 
						rate3_device                
						);

					cudaThreadSynchronize();


					//			debugDumpMatrix(rate1_device,"rate 1", exerciseIndices_vec.size(),n_vectors);
					// now for cash flows

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
						basisFunctionVariables_gold_cube.Facade()// output location
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

					//			debugDumpMatrix(genFlows1_cpu_matrix,"genFlows1_cpu_matrix");
					//			debugDumpMatrix(genFlows2_cpu_matrix,"genFlows2_cpu_matrix");

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
						fromGenIndicesToExerciseDeflations_vec, // rate time index to discount to 
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
						fromGenIndicesToExerciseDeflations_vec, // rate time index to discount to 
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
						fromGenIndicesToExerciseDeflations_device, 
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
						fromGenIndicesToExerciseDeflations_device, 
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
						stepToExerciseIndices_device ,
						firstPositiveStep);

					AggregateFlows_main(
						aggregatedFlows_device,// output added to, not overwritten
						totalPaths,
						exerciseIndices_vec.size(), 
						discounted_flows2_device,
						pathsPerBatch, 
						pathsPerBatch*batchCounter,
						numberSteps, 
						stepToExerciseIndices_device ,
						firstPositiveStep
						);


					cudaThreadSynchronize();

					totalTimeForCashFlowAggregationGPU += h12.timePassed();




				}//  end of   for (int batchCounter=0; batchCounter < numberOfBatches; ++batchCounter)

				std::cout << " finished loop...\n";


				std::cout << "Evolved rates \n";

				//   }// inner scope for destruction basisVariableExample_gold:

				std::cout << " exited inner scope\n";

				// now it's time to actually test!

				// first, we do the basis variables 
				thrust::host_vector<float> final_basis_variables_gpu_host(final_basis_variables_device);
				CubeConstFacade<float> final_basis_variables_gpu_cube(&final_basis_variables_gpu_host[0],exerciseIndices_vec.size(),basisVariableExample_gold<float>::maxVariablesPerStep( ), totalPaths);

				int mismatches=0;

				std::cout << " entering test loop for extraction\n";

				std::vector<int> basisVariablesPerIndex(exerciseIndices_vec.size());
				for (int s=0; s < static_cast<int>(exerciseIndices_vec.size()); ++s)
					basisVariablesPerIndex[s]=  basisVariableExample_gold<float>::actualNumberOfVariables(exerciseIndices_vec[s], aliveIndex[exerciseIndices_vec[s]],number_rates);


				for (int p=0; p < totalPaths; ++p)
				{
					for (int s=0; s < static_cast<int>(exerciseIndices_vec.size()); ++s)
					{
						for (int v=0; v < basisVariablesPerIndex[s]; ++v)
						{                   
							double x= final_basis_variables_gpu_cube(s,v,p);
							double y = basisFunctionVariables_gold_cube(s,v,p);
							if ( fabs( x-y ) > 100*tolerance)
							{
								++mismatches;
								std::cout <<" basis mismatch "  << p << ","<< s << ","<<v << ","<<x << ","<<y<< "\n";
							}
						}
					}
				}



				std::cout << " exiting test loop\n";

				if (mismatches>0)
				{
					std::cout << "Basis extraction failed\n";
					std::cout << "Number of mismatches " << mismatches << " out of "<<totalPaths*numberOfExerciseDates*basisVariableExample_gold<float>::maxVariablesPerStep( ) << "\n";

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
					deflation_locations_vec,
					some_numeraireValues_vec //output
					);


				thrust::device_vector<float> some_numeraireValues_device(exerciseIndices_vec.size()*totalPaths);

				spot_measure_numeraires_extraction_main(  spot_measure_values_device,
					some_numeraireValues_device, //output
					totalPaths,
					totalPaths,
					0,
					numberSteps,
					deflation_locations_vec.size(), 
					deflation_locations_device
					);

				std::vector<double> aggregatedFlows_realv_vec(aggregatedFlows_vec.size());
				std::copy(aggregatedFlows_vec.begin(),aggregatedFlows_vec.end(),aggregatedFlows_realv_vec.begin());

				MatrixFacade<double> aggregatedFlows_cpu_realv_matrix(&aggregatedFlows_realv_vec[0],
					aggregatedFlows_cpu_matrix.rows(),
					aggregatedFlows_cpu_matrix.columns());

				std::vector<double> exercise_values_realv_vec(exercise_values_vec.size());
				std::copy(exercise_values_vec.begin(),exercise_values_vec.end(),exercise_values_realv_vec.begin());

				MatrixFacade<double> exercise_values_realv_matrix(&exercise_values_realv_vec[0],
					exercise_values_matrix.rows(),
					exercise_values_matrix.columns());

				std::vector<double> exercise_values_zero_realv_vec(exercise_values_vec.size());
				std::fill(exercise_values_zero_realv_vec.begin(),exercise_values_zero_realv_vec.end(),0.0);

				MatrixFacade<double> exercise_values_zero_realv_matrix(&exercise_values_zero_realv_vec[0],
					exercise_values_matrix.rows(),
					exercise_values_matrix.columns());

				thrust::device_vector<float> exercise_values_zero_float_device(exercise_values_vec.size(),0.0f);


				std::vector<double> some_numeraireValues_realv_vec(some_numeraireValues_vec.size());
				std::copy(some_numeraireValues_vec.begin(),some_numeraireValues_vec.end(),some_numeraireValues_realv_vec.begin());

				MatrixFacade<double> some_numeraireValues_realv_matrix(&some_numeraireValues_realv_vec[0],
					some_numeraireValues_matrix.rows(),
					some_numeraireValues_matrix.columns());

				int maxBasisVariables = basisVariableExample_gold<float>::maxVariablesPerStep( );

				int numberBasisFunctions = 2*maxBasisVariables+1;
				if (useCrossTerms)
					numberBasisFunctions = quadraticPolynomialCrossDevice::functionValues(maxBasisVariables);


				// start doing regressions


				std::vector<float> variableShiftsToSubtract_vec(numberOfExerciseDates*(numberOfExtraRegressions+1)*maxBasisVariables,0.0f);
				std::vector<float> variableDivisors_vec(numberOfExerciseDates*(numberOfExtraRegressions+1)*maxBasisVariables,1.0f);

				CubeFacade<float> variableShiftsToSubtract_cube(variableShiftsToSubtract_vec,numberOfExerciseDates,numberOfExtraRegressions+1,maxBasisVariables);
				CubeFacade<float> variableDivisors_cube(variableDivisors_vec,numberOfExerciseDates,numberOfExtraRegressions+1,maxBasisVariables);


				if (globallyNormalise)
				{
					int layerSize = totalPaths*maxBasisVariables;
					// let's rescale our basis variables to improve stability
					for (int i=0; i < numberOfExerciseDates; ++i)
					{
						for (int j=0; j < basisVariablesPerIndex[i]; ++j)
						{

							thrust::device_vector<float>::iterator start =final_basis_variables_device.begin()+i*layerSize + j*totalPaths;
							thrust::device_vector<float>::iterator end = start+totalPaths;
							float sum = thrust::reduce(start,end);
							float mean = sum/totalPaths;
							float sumsq = thrust::transform_reduce(start,end,squareof(),0.0f,thrust::plus<float>());					
							float sd = sqrt( sumsq/(totalPaths) - mean*mean);
							thrust::transform(start,end,start,shiftAndMult(-mean,1.0f/sd));

							for (int k=0; k < numberOfExtraRegressions+1; ++k)
							{
								variableShiftsToSubtract_cube(i,k,j) = mean;
								variableDivisors_cube(i,k,j) = sd;
							}
						}
					}
				}





				std::vector<double> products_cube_vec;
				std::vector<double> targets_mat_vec;
				std::vector<double> productsM_cube_vec;
				std::vector<double> targetsM_mat_vec;


				std::vector<double> ls_coefficients_multi_vec(maxRegressionDepth*exerciseIndices_vec.size()*numberBasisFunctions);
				CubeFacade<double> regression_coefficients_cube(ls_coefficients_multi_vec,exerciseIndices_vec.size(),
					maxRegressionDepth,numberBasisFunctions);

				Matrix_gold<double> lowerCuts_mat(exerciseIndices_vec.size(),maxRegressionDepth,0.0);
				Matrix_gold<double> upperCuts_mat(exerciseIndices_vec.size(),maxRegressionDepth,0.0);


				Cube_gold<double> means_cube_gold(exerciseIndices_vec.size(),numberOfExtraRegressions+1, basisVariableExample_gold<float>::maxVariablesPerStep( ),0.0);
				Cube_gold<double> sds_cube_gold(exerciseIndices_vec.size(),numberOfExtraRegressions+1, basisVariableExample_gold<float>::maxVariablesPerStep( ),0.0);
				// change types to double

				std::vector<float> final_basis_variables_vec(stlVecFromDevVec(final_basis_variables_device));

				CubeFacade<float> basisFunctionVariables_cube(&final_basis_variables_vec[0],exerciseIndices_vec.size(),basisVariablesPerStep,totalPaths);

				std::vector<double> basisFunctionVariables_realv_vec(final_basis_variables_vec.size());
				std::copy(final_basis_variables_vec.begin(),final_basis_variables_vec.end(),basisFunctionVariables_realv_vec.begin());

				CubeConstFacade<double> basisFunctionVariables_realv_cube(&basisFunctionVariables_realv_vec[0],
					basisFunctionVariables_cube.numberLayers(),
					basisFunctionVariables_cube.numberRows(),
					basisFunctionVariables_cube.numberColumns());

				//			debugDumpMatrix(some_numeraireValues_vec,"some_numeraireValues_vec",some_numeraireValues_realv_matrix.rows(),
				//			some_numeraireValues_realv_matrix.columns());
				
				bool normalise = false; 


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
					basisVariableExample_gold<float>::maxVariablesPerStep( ),
					aggregatedFlows_cpu_realv_matrix, // deflated to current exercise time
					exercise_values_zero_realv_matrix, // deflated to current exercise time
					some_numeraireValues_realv_matrix,
					//         deflation_locations_vec,
					totalPaths,
					numberOfExtraRegressions,
					lowerPathCutoff,
					regressionSelector,
					useCrossTerms);
				std::cout << " LSEstM, " << lsestM << "\n";

	//			debugDumpCube(regression_coefficients_cube,"regression_coefficients_cube");

				std::vector<double> AndersenShifts_vec(exerciseIndices_vec.size());
				std::fill(AndersenShifts_vec.begin(),AndersenShifts_vec.end(),0.0);



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
						basisVariableExample_gold<float>::maxVariablesPerStep( ),
						aggregatedFlows_cpu_realv_matrix, 
						exercise_values_zero_realv_matrix, 
						some_numeraireValues_realv_matrix,
						totalPaths,
						//		   deflation_locations_vec,
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
						basisVariableExample_gold<float>::maxVariablesPerStep( ),
						aggregatedFlows_cpu_realv_matrix, 
						exercise_values_zero_realv_matrix, 
						some_numeraireValues_realv_matrix,
						totalPaths,
						//		   deflation_locations_vec,
						strategy
						);

				}



				if (fabs(lsestM-lsest2M )> 5e-5)
				{
					std::cout << " forwards and backwards multi LS gold estimates don't agree\n";
				}
				else 
					--numberFails;

				std::cout <<"  LSEst2M, " << lsest2M << "\n";

				if (globallyNormalise && !normalise)
				{
					means_cube_gold=CubeTypeConvert<double,float>(Cube_gold<float>(variableShiftsToSubtract_cube));
					sds_cube_gold=CubeTypeConvert<double,float>(Cube_gold<float>(variableDivisors_cube));

				}



				// also test generation of polynomials from variables

				int totalNumberBasisFunctions = useCrossTerms ? totalPaths*(3*basisVariableExample_gold<float>::maxVariablesPerStep( )+1) : totalPaths*(2*basisVariableExample_gold<float>::maxVariablesPerStep( )+1);


				thrust::host_vector<int> basisVariablesEachStep_host(basisVariablesPerIndex.begin(),basisVariablesPerIndex.end());
				thrust::device_vector<int> basisVariablesEachStep_device(basisVariablesEachStep_host);

				thrust::device_vector<int> basisVariableInt_data_device(numberOfExerciseDates,0);
				thrust::device_vector<float> basisVariableFloat_data_device(numberOfExerciseDates,0.0f);
				//	thrust::device_vector<float> coefficients_device(numberOfExerciseDates*numberBasisFunctions);
				//		std::vector<Realv> coefficients_vec(coefficients_device.size());


				std::vector<float> products_gpu_cube_vec(numberOfExerciseDates*numberBasisFunctions*numberBasisFunctions);
				std::vector<float> targets_gpu_mat_vec(numberOfExerciseDates*numberBasisFunctions);

				//		CubeConstFacade<double> products_gold_cube(products_cube_vec,numberExerciseDates,numberBasisFunctions,numberBasisFunctions);
				//			MatrixConstFacade<double>	targets_gold_mat(targets_mat_vec,numberExerciseDates,numberBasisFunctions);
				//	MatrixConstFacade<double> coefficients_gold(ls_coefficients,numberExerciseDates,numberBasisFunctions);



				thrust::device_vector<float> lowercuts_device(numberOfExerciseDates*(numberOfExtraRegressions+1));
				thrust::device_vector<float> uppercuts_device(numberOfExerciseDates*(numberOfExtraRegressions+1));


				thrust::device_vector<float> coefficients_M_device(numberOfExerciseDates*(numberOfExtraRegressions+1)*maxBasisVariables);
				std::vector<double> coefficients_M_vec(coefficients_M_device.size());

				//	RegressionSelectorStandardDeviations regressionSelector(sdsForCutOff);

				RegressionSelectorFraction regressionSelector_gpu(lowerFrac, upperFrac,initialSDguess, multiplier);

    	LinearSolverHandCraftedAndGold solve(1,1,0,0);


         
            double LSest3M  
                = generateRegressionCoefficientsViaLSMultiquadratic_flexi_main(numberOfExerciseDates,
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
					//	deflation_locations_vec,
					totalPaths,
					useCrossTerms,
					numberOfExtraRegressions,
					regressionSelector_gpu,
					lowerPathCutoff,solve);

	//			debugDumpCube(coefficients_M_vec,"coefficients_M_vec", numberOfExerciseDates,(numberOfExtraRegressions+1),numberBasisFunctions);


				std::cout << " backwards estimate using multi regression LS cuda code " << LSest3M << "\n";

				if (fabs(LSest3M - lsestM)> lsfirstPassTol)
				{
					std::cout << " CPU and GPU first pass values disagree: test failed.\n";
				}
				else
					--numberFails;



				// first passes are done
				// we still need to do the second passes!


				std::vector<float> batchValuesMGold(numberOfSecondPassBatches);
				std::vector<float> batchValuesGPU_vec(numberOfSecondPassBatches);

				thrust::device_vector<float> batch_basis_variables_device(pathsPerSecondPassBatch*exerciseIndices_vec.size()*maxBasisVariables);

				thrust::device_vector<float> discountedFlows_dev(pathsPerSecondPassBatch*numberSteps);
				thrust::device_vector<float> summedDiscountedFlows1_dev(pathsPerSecondPassBatch);
				thrust::device_vector<float> summedDiscountedFlows2_dev(pathsPerSecondPassBatch);
              
                for (int pass=0; pass < 2; pass++)
				{

					std::vector<double> coefficientsMultiToUse_vec( pass ==0 ? coefficients_M_vec : ls_coefficients_multi_vec );

					Matrix_gold<double> lowerCutsToUse_mat(pass ==0 ? MatrixCastDeviceVec<double,float>(lowercuts_device,numberOfExerciseDates,maxRegressionDepth) : lowerCuts_mat);
					Matrix_gold<double> upperCutsToUse_mat(pass ==0 ? MatrixCastDeviceVec<double,float>(uppercuts_device,numberOfExerciseDates,maxRegressionDepth) : upperCuts_mat);


					//				thrust::device_vector<float> lowerCutsToUse_device(pass ==0 ?  lowercuts_device :lowerCuts_mat.getDataVector());
					//				thrust::device_vector<float> upperCutsToUse_device(pass ==0 ?  uppercuts_device :upperCuts_mat.getDataVector());

					if (pass==0)
						std::cout << " \n\nDoing second pass using GPU generation for ls coefficients  \n";
					else
						std::cout << " \n\nDoing second pass using  using CPU generation for ls coefficients \n";


					int floatDataCrossSize;
					int intDataCrossSize;


					LSAMultiExerciseStrategyQuadraticCross_gpu::outputDataVectorSize(floatDataCrossSize,
						intDataCrossSize, 
						numberOfExtraRegressions+1,
						exerciseIndices_vec.size(),
						basisVariablesPerIndex);


					thrust::device_vector<float>  exerciseStrategyDataCrossFloat_device( floatDataCrossSize);                                 
					thrust::device_vector<int>  exerciseStrategyDataCrossInt_device( intDataCrossSize);     

					LSAMultiExerciseStrategyQuadraticCross_gpu::outputDataVectors(exerciseStrategyDataCrossFloat_device.begin(), 
						exerciseStrategyDataCrossInt_device.begin(),
						numberOfExtraRegressions+1,
						static_cast<int>(exerciseIndices_vec.size()),
						basisVariablesPerIndex ,
						AndersenShifts_vec,
						variableShiftsToSubtract_vec,
						variableDivisors_vec,
						lowerCutsToUse_mat.getDataVector(),
						upperCutsToUse_mat.getDataVector(),
						coefficientsMultiToUse_vec);



					int floatDataNonCrossSize;
					int intDataNonCrossSize;


					LSAMultiExerciseStrategyQuadratic_gpu::outputDataVectorSize(floatDataNonCrossSize,
						intDataNonCrossSize, 
						numberOfExtraRegressions+1,
						exerciseIndices_vec.size(),
						basisVariablesPerIndex);


					thrust::device_vector<float>  exerciseStrategyDataNonCrossFloat_device( floatDataNonCrossSize);                                 
					thrust::device_vector<int>  exerciseStrategyDataNonCrossInt_device( intDataNonCrossSize);     

					LSAMultiExerciseStrategyQuadratic_gpu::outputDataVectors(exerciseStrategyDataNonCrossFloat_device.begin(), 
						exerciseStrategyDataNonCrossInt_device.begin(),
						numberOfExtraRegressions+1,
						static_cast<int>(exerciseIndices_vec.size()),
						basisVariablesPerIndex ,
						AndersenShifts_vec,
						variableShiftsToSubtract_vec,
						variableDivisors_vec,
						lowerCutsToUse_mat.getDataVector(),
						upperCutsToUse_mat.getDataVector(),
						coefficientsMultiToUse_vec);



					thrust::host_vector<int> exerciseIndices_host(exerciseIndices_vec.begin(),exerciseIndices_vec.end());
					thrust::device_vector<int> exerciseIndices_device(exerciseIndices_host);

					thrust::device_vector<float> exerciseValueDataFloatdummy_device(1,0.0f);
					thrust::device_vector<int> exerciseValueDataIntdummy_device(1,0);

					thrust::device_vector<float> estCValues_device(pathsPerSecondPassBatch*numberOfExerciseDates);
					thrust::device_vector<int> exerciseTimeIndices_device(pathsPerSecondPassBatch);



                    
					for (int k=0; k < numberOfSecondPassBatches; ++k)
					{
						int path_offset= numberOfBatches * pathsPerBatch*duplicate+k*pathsPerSecondPassBatch;
						LMMEvolutionRoutineRawSingleKernel(pathsPerBatch, 
							path_offset, 
							number_rates, 
							factors, 
							number_rates, 
							n_poweroftwo,
							scrambler_device, 
							A_device,
							drifts_device, 
							displacements_device,
							rates_device, 
							logRates_device, 
							taus_device, 
							initial_drifts_device, 
							aliveIndex, 
							alive_device, 
							// buffers
							SobolInts_buffer_device, 
							device_input, 
							dev_output, 
							thrust::raw_pointer_cast(&dev_correlated_rates[0]), 
							thrust::raw_pointer_cast(&e_buffer_device[0]),
							thrust::raw_pointer_cast(&e_pred_buffer_device[0]),
							thrust::raw_pointer_cast(&evolved_rates_device[0]),
							thrust::raw_pointer_cast(&evolved_log_rates_device[0]),
							thrust::raw_pointer_cast(&discounts_device[0]),
							useSharedMem,
							doDiscounts,
							newBridge,
							fermiArch,
							0 // threads
							);


						cudaThreadSynchronize();

            

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

						forward_rate_extraction_steps_gpu_main(  evolved_rates_device, 
							extractor1,
							exerciseIndices_vec,
							n_vectors,
							number_rates, 
							rate1_device                
							);

						forward_rate_extraction_steps_gpu_main(  swap_rates_device, 
							extractor2,
							exerciseIndices_vec,
							n_vectors,
							number_rates, 
							rate2_device                
							);

						forward_rate_extraction_steps_gpu_main(  annuities_device, 
							extractor3,
							exerciseIndices_vec,
							n_vectors,
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

						earlyExerciseNullGold<float> exerciseValue;

						CubeConstFacade<float> forwards2_cube(&evolved_rates_gpu_vec[0],numberSteps,numberOfRates,pathsPerSecondPassBatch);

						std::vector<float> accruals1_nulled_gold(accruals1);
						std::vector<float> accruals2_nulled_gold(accruals2);
						for (int i=0; i < numberNonCallDates;++i)
						{
							accruals1_nulled_gold[i] =0.0f;
							accruals2_nulled_gold[i] =0.0f;
						}

						Swap_gold<float> product( numberSteps, strike,payReceive,  accruals1_nulled_gold, accruals2_nulled_gold );

						std::vector<bool> exerciseIndicatorsbool_vec(exerciseIndicators_vec.size());
						for (size_t i=0; i < exerciseIndicators_vec.size(); ++i)
							exerciseIndicatorsbool_vec[i] = exerciseIndicators_vec[i] != 0;




						// multi gold version
						std::vector<float> flow1_multi_vec(pathsPerSecondPassBatch*numberOfRates);
						std::vector<float> flow2_multi_vec(pathsPerSecondPassBatch*numberOfRates);

						if (useCrossTerms)
						{
                            
							std::vector<quadraticPolynomialCrossGenerator> functionProducers;

							for (size_t i=0; i < exerciseIndices_vec.size(); ++i)
								functionProducers.push_back(quadraticPolynomialCrossGenerator(basisVariablesPerIndex[i]));

           
							LSAMultiExerciseStrategy<quadraticPolynomialCrossGenerator,float> exMStrategy(
                                static_cast<int>(exerciseIndices_vec.size()), //int numberExerciseTimes_,
	           				    basisVariablesPerIndex, //    const std::vector<int>& variablesPerStep_,
								AndersenShifts_vec,//    const std::vector<float>& basisShifts_,
								coefficientsMultiToUse_vec, //  const std::vector<float>& basisWeights_,
								maxRegressionDepth, //   	int regressionDepth_,
								lowerCutsToUse_mat.ConstFacade(), //	const MatrixConstFacade<float>& lowerCuts_,
								upperCutsToUse_mat.ConstFacade(), //const MatrixConstFacade<float>& upperCuts_,
								means_cube_gold.ConstFacade(),//	const CubeConstFacade<float>& means_cube_,
								sds_cube_gold.ConstFacade(), //	const CubeConstFacade<float>& sds_cube_,
								functionProducers, //  const std::vector<T>& functionProducers_,
                                maxBasisVariables
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

							LSAMultiExerciseStrategy<quadraticPolynomialGenerator,float> exMStrategy(static_cast<int>(exerciseIndices_vec.size()),
								basisVariablesPerIndex,
								AndersenShifts_vec,
								coefficientsMultiToUse_vec, 
								maxRegressionDepth,
								lowerCutsToUse_mat.ConstFacade(),
								upperCutsToUse_mat.ConstFacade(),
								means_cube_gold.ConstFacade(),
								sds_cube_gold.ConstFacade(),
								functionProducers,
                                maxBasisVariables
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

						//				debugDumpMatrix(flow1_multi_vec,"flow1_multi_vec",numberSteps,pathsPerSecondPassBatch);
						//			debugDumpMatrix(flow2_multi_vec,"flow2_multi_vec",numberSteps,pathsPerSecondPassBatch);


						//		debugDumpVector(summedDiscountedFlows1_M_vec,"summedDiscountedFlows1_M_vec");
						//		debugDumpVector(summedDiscountedFlows2_M_vec,"summedDiscountedFlows2_M_vec");

						float batchSumMGold = std::accumulate(summedDiscountedFlows1_M_vec.begin(),summedDiscountedFlows1_M_vec.end(),0.0f)
							+std::accumulate(summedDiscountedFlows2_M_vec.begin(),summedDiscountedFlows2_M_vec.end(),0.0f);

						batchValuesMGold[k] = batchSumMGold/ pathsPerSecondPassBatch;





						if(useCrossTerms)
						{

							cashFlowGeneratorEE_main<Swap ,earlyExerciseNull ,LSAMultiExerciseStrategyQuadraticCross_gpu >(
								genFlows1_device,//  thrust::device_vector<float>&  genFlows1_device, 
								genFlows2_device,  //        thrust::device_vector<float>&  genFlows2_device, 
								aux_swap_data_2_device, // productData_device
								exerciseValueDataFloatdummy_device, //exerciseValueDataFloat_device 
								exerciseValueDataIntdummy_device, //exerciseValueDataInt_device
								exerciseStrategyDataCrossFloat_device, //exerciseStrategyDataFloat_device
								floatDataCrossSize,                    //     int exerciseStrategyDataFloat_size,
								exerciseStrategyDataCrossInt_device, // exerciseStrategyDataInt_device
								intDataCrossSize,                            //exerciseStrategyDataInt_size
								pathsPerSecondPassBatch,  // paths
								numberSteps,                // numberSteps
								static_cast<int>(exerciseIndices_vec.size()),      //numberExerciseDates
								exerciseIndicatorsBool_device,
								rate1_cf_device, 
								rate2_cf_device, 
								rate3_cf_device, 
								batch_basis_variables_device,
								maxBasisVariables,
								evolved_rates_device, 
								discounts_device,
								estCValues_device, 
								exerciseTimeIndices_device);
						}
						else
						{
							cashFlowGeneratorEE_main<Swap ,earlyExerciseNull ,LSAMultiExerciseStrategyQuadratic_gpu >(
								genFlows1_device,
								genFlows2_device, 
								aux_swap_data_2_device,
								exerciseValueDataFloatdummy_device,
								exerciseValueDataIntdummy_device,
								exerciseStrategyDataNonCrossFloat_device,
								floatDataNonCrossSize,
								exerciseStrategyDataNonCrossInt_device,
								intDataNonCrossSize,
								pathsPerSecondPassBatch, 
								numberSteps,
								static_cast<int>(exerciseIndices_vec.size()),
								exerciseIndicatorsBool_device,
								rate1_cf_device, 
								rate2_cf_device, 
								rate3_cf_device, 
								batch_basis_variables_device,
								maxBasisVariables,
								evolved_rates_device, 
								discounts_device,
								estCValues_device, 
								exerciseTimeIndices_device);
						}
						CUT_CHECK_ERR("cashFlowGeneratorEE_main failed \n");

						//				debugDumpVector(exerciseTimeIndices_device,"exerciseTimeIndices_device");
						//			debugDumpMatrix(estCValues_device,"estCValues_device",numberOfExerciseDates,pathsPerSecondPassBatch);



						cashFlowDiscounting_gpu_main(firstIndex_device, 
							secondIndex_device,
							thetas_device, 
							discounts_device, 
							genFlows1_device, 
							spot_measure_values_2_device,
							pathsPerSecondPassBatch, 
							numberSteps, 
							useTextures,
							useShared,
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
							useTextures,
							useShared,
							discountedFlows_dev, // output
							summedDiscountedFlows2_dev); // output

						//			debugDumpMatrix(genFlows1_device,"genFlows1_device",numberSteps,pathsPerSecondPassBatch);
						//		debugDumpMatrix(genFlows2_device,"genFlows2_device",numberSteps,pathsPerSecondPassBatch);


						float batchValueGPU = thrust::reduce(summedDiscountedFlows1_dev.begin(),summedDiscountedFlows1_dev.end());
						batchValueGPU += thrust::reduce(summedDiscountedFlows2_dev.begin(),summedDiscountedFlows2_dev.end());
						batchValueGPU /= pathsPerSecondPassBatch;

						batchValuesGPU_vec[k] = batchValueGPU;



					}
                    
					float valueMGold = std::accumulate(batchValuesMGold.begin(),batchValuesMGold.end(),0.0f)/numberOfSecondPassBatches;


					std::cout << " second pass  gold multi value, " << valueMGold << "\n";

					std::cout << " second pass cpu values :\n";

					for (size_t i=0; i < batchValuesMGold.size(); ++i)
						std::cout << batchValuesMGold[i] << ",";
					std::cout << "\n";

					float valueMGpu = std::accumulate(batchValuesGPU_vec.begin(),batchValuesGPU_vec.end(),0.0f)/numberOfSecondPassBatches;


					std::cout << " second pass  gpu multi value, " << valueMGpu << "\n";

					std::cout << " second pass gpu values :\n";

					for (size_t i=0; i < batchValuesGPU_vec.size(); ++i)
						std::cout << batchValuesGPU_vec[i] << ",";
					std::cout << "\n";

					if (fabs(valueMGold - valueMGpu) < 1e-4)
					{
						--numberFails;

						std::cout << " second pass test passes\n";
					}
					else
						std::cout << " gold and GPU code disagree on second pass.\n";

                     
				}
			}

		} // outer scope for destruction
	}
	cudaThreadExit();

	return numberFails;
}
