//
//
//                  LS_main.h
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef LS_MAIN_H
#define LS_MAIN_H
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <vector>
#include <gold/MatrixFacade.h>
#include <float.h>
#include <gold/math/fp_utilities.h>
#include <gold/math/typedefs_math_gold.h>
#include <gold/math/svd_gold.h>
#include <iostream>
#include <LS_gpu.h>
#include <outerProduct_gpu.h>
#include <output_device.h>
#include <time.h>
//#include <cuBlas.h>
 #include <thrust/functional.h>
#include <gold/Timers.h>


void expandBasisFunctions_quadratic_main(
	int totalNumberOfPaths, 
	thrust::device_vector<float>& basisVariables_device, // input location for basis variables for all steps,
	int exerciseNumber,
	int maxBasisVariableSize, 
	int variableSizeThisStep,
	bool useCrossTerms,
	thrust::device_vector<float>& basisFunctions_device
	);


// compute estimated continuation value using regression coefficients
// and then subtract exercise value
// return value is time taken
double est_continuation_net_value_evaluation_main(int pathsRemaining,
												  int exerciseNumber,
												  int maxNumberBasisFunctions,
												  int maxNumberOfRegressions,
												  int whichRegression,
												  thrust::device_vector<float>& basisFunctionsRemaining_device, // just the function values for this step
												  int numberBasisFunctions,
												  thrust::device_vector<float>& basis_weights_device, // the weights for all steps all regressions
												  thrust::device_vector<float>& exerciseValues_device,      // the exerciseValues for all steps deflated to  exercise times
												  float AndersenShift,
												  thrust::device_vector<float>& output_device);
/*
double oneStepUpdateLS_main(int paths,
							int exerciseNumber,
							thrust::device_vector<float>& basisFunctions_global, // just the function values for this step
							int numberBasisFunctions,
							thrust::device_vector<float>& basis_weights_global, // the weights for all steps
							thrust::device_vector<float>& continuationValues_global,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
							thrust::device_vector<float>& deflatedCashValues_global,  //the cash flow values for all steps deflated to exercise times
							thrust::device_vector<float>& exerciseValues_global,      // the exerciseValues for all steps deflated to exercise times
							float AndersenShift);
							*/
double  oneStepUpdateLS_multi_main(int paths,
								   int exerciseNumber,
								   thrust::device_vector<float>& basisFunctions_device, // just the function values for this step
								   int numberBasisFunctions,
								   int basisFunctionsRowSize,
								   int maxRegressionDepth,
								   thrust::device_vector<float>& basis_weights_device, // the weights for all steps and regression depths, 
								   thrust::device_vector<float>& lowerCuts_device,
								   thrust::device_vector<float>& upperCuts_device,
								   thrust::device_vector<float>& continuationValues_device,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
								   thrust::device_vector<float>& deflatedCashFlows_device,   // the cash-flow values all steps deflated to ex times
								   thrust::device_vector<float>& exerciseValues_device,      // the exerciseValues for all steps deflated to  exercise times
								   float AndersenShift,
								   thrust::device_vector<float>& estContValues_device,
								   thrust::device_vector<int>& exercises_device);

// returns time taken
// overwrites deflatedNextStepValues_global
// numeraireValues_global is matrix of form exercise dates \times paths 


double updateDeflationOfContinuationValues_main(  thrust::device_vector<float>& deflatedNextStepValues_device,  
												thrust::device_vector<float>& numeraireValues_device,
												int paths,
												int newStepNumber,
												int oldStepNumber);
/*
double generateRegressionCoefficientsViaLSquadratic_gpu(int numberExerciseDates,
					//									std::vector<float>& products_cube_vec,
				//										std::vector<float>& targets_mat_vec,
                                                        thrust::device_vector<int>& basisVariableInt_data_device,
                                                        thrust::device_vector<float>& basisVariableFloat_data_device,
                                                        thrust::device_vector<float>& coefficients_device, // the LS coefficients are placed here 
                                                        std::vector<Realv>& coefficients_vec, // the LS coefficients are also placed here 
                                                        thrust::device_vector<float>&  basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
                                                        thrust::device_vector<int>& basisVariablesEachStep_device, //vector of the number of basis variables for each step
                                                        const std::vector<int>& basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
                                                        int maxBasisVariables,
                                                        thrust::device_vector<float>& deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
                                                        thrust::device_vector<float>& deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
                                                        thrust::device_vector<float>& numeraireValues_device, // numeraire vals for each, matrix of times and paths 
														const std::vector<int>& deflation_locations_vec,
                                                        int paths,
                                                        int threads,
                                                        int blocks,
														bool normalise,
														bool useCrossTerms,
											   bool useTestMode,
											   bool &testPassed,
											   const CubeConstFacade<double>& products_gold_cube, // ignored if not in testmode
											   const MatrixConstFacade<double>& targets_gold_mat,// ignored if not in testmode
											   const MatrixConstFacade<double>& coefficients_gold,                                 // ignored if not in testmode
											   double testTolerance   // ignored if not in testmode		
                                                        );

// this one doesn't give possibility of using test mode
double generateRegressionCoefficientsViaLSquadratic_gpu(int numberExerciseDates,
                                                        thrust::device_vector<int>& basisVariableInt_data_device,
                                                        thrust::device_vector<float>& basisVariableFloat_data_device,
                                                        thrust::device_vector<float>& coefficients_device, // the LS coefficients are placed here 
                                                        std::vector<Realv>& coefficients_vec, // the LS coefficients are also placed here 
                                                        thrust::device_vector<float>&  basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
                                                        thrust::device_vector<int>& basisVariablesEachStep_device, //vector of the number of basis variables for each step
                                                        const std::vector<int>& basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
                                                        int maxBasisVariables,
                                                        thrust::device_vector<float>& deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
                                                        thrust::device_vector<float>& deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
                                                        thrust::device_vector<float>& numeraireValues_device, // numeraire vals for each, matrix of times and paths 
                                                        const std::vector<int>& deflation_locations_vec,
														int paths,
                                                        int threads,
                                                        int blocks,
														bool normalise,
														bool useCrossTerms);
*/

struct square
{
__host__ __device__
float operator()(float x)
{
return x * x;
}
};
 

struct scaleAndShift
{
	float s;
	float a;
__host__ __device__
float operator()(float x)
{
return s * x+a;
}
};
/*
template<class T>
double generateRegressionCoefficientsViaLS_gpu(int numberExerciseDates,
			//								   std::vector<float>& products_cube_vec,
			//								   std::vector<float>& targets_mat_vec,
											   thrust::device_vector<int>& basisVariableInt_data_device,
											   thrust::device_vector<float>& basisVariableFloat_data_device,
											   thrust::device_vector<float>& coefficients_device, // the LS coefficients are placed here 
											   std::vector<Realv>& coefficients_vec, // the LS coefficients are also placed here 
											   thrust::device_vector<float>&  basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
											   thrust::device_vector<int>& basisVariablesEachStep_device, //vector of the number of basis variables for each step
											   const std::vector<int>& basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
											   int maxBasisVariables,
											   thrust::device_vector<float>& deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
											   thrust::device_vector<float>& deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
											   thrust::device_vector<float>& numeraireValues_device, // numeraire vals for each, matrix of times and paths 
											   const std::vector<int>& deflation_locations_vec,
											   int paths,
											   int threads,
											   int blocks,
											   bool normalise,
											   bool useTestMode,
											   bool &testPassed,
											   const CubeConstFacade<double>& products_gold_cube, // ignored if not in testmode
											   const MatrixConstFacade<double>& targets_gold_mat,// ignored if not in testmode
											   const MatrixConstFacade<double>& coefficients_gold,                                 // ignored if not in testmode
											   double testTolerance   // ignored if not in testmode										  
											   )
{
	double timeInCuBlas=0.0;
	
	int numberFailures =0;
	cublasInit();
	int numberDataPoints= T::functionValues(maxBasisVariables);
	thrust::device_vector<float> basisFunctions_workspace_device(numberDataPoints  * paths );
	thrust::device_vector<float> basisFunctions_workspace2_device(numberDataPoints  * paths );
	thrust::device_vector<float> basisExpectation_matrix_device(numberDataPoints  * numberDataPoints );
	thrust::device_vector<float> basisExpectation_workspace_device(blocks);

	thrust::device_vector<float> basisproducts_device(numberDataPoints   );

	thrust::host_vector<double> basisExpectation_matrix_host(basisExpectation_matrix_device.size());
	thrust::host_vector<double> basisproducts_host(basisproducts_device.size());


	thrust::device_vector<float> continuationValues(paths,0.0f);

	int oldnumberBasisVariables = *basisVariablesEachStep_vec.rbegin();


	SVDSquareMatrix svdSolver(T::functionValues(*basisVariablesEachStep_vec.rbegin()));

	coefficients_vec.resize(numberDataPoints*numberExerciseDates);
	thrust::host_vector<float> coefficients_host(coefficients_vec.size());
	std::vector<double> coefficients_step_vec(numberDataPoints);

	MatrixFacade<double> regressCoeffs_mat(&coefficients_vec[0],numberExerciseDates,numberDataPoints);
	MatrixFacade<float> regressCoeffs_host_mat(&coefficients_host[0],numberExerciseDates,numberDataPoints);

//	CubeFacade<float> products_cube(products_cube_vec,numberExerciseDates,numberDataPoints,numberDataPoints); 
//	MatrixFacade<float> targets_mat(targets_mat_vec,numberExerciseDates,numberDataPoints);

	std::vector<float> means_vec(numberDataPoints,0.0f);
    std::vector<float> sds_vec(numberDataPoints,1.0f);


	//   std::cout << clock()<<", start,"; 
	for (int step= numberExerciseDates-1;  step >=0;--step)
	{
		int numberBasisVariables = basisVariablesEachStep_vec[step];
		int numberDataPoints = T::functionValues(numberBasisVariables);

		if (numberBasisVariables != oldnumberBasisVariables)
		{
			SVDSquareMatrix tmp(numberDataPoints);
			svdSolver=tmp;

			means_vec.resize(numberDataPoints);
			sds_vec.resize(numberDataPoints);


		}
	

		expandBasisFunctions<T>( thrust::raw_pointer_cast(& basisVariableInt_data_device[0]), // to access texture containing data specific to the basis variables
			thrust::raw_pointer_cast(&basisVariableFloat_data_device[0]) ,                  
			paths, 
			thrust::raw_pointer_cast(&basisFunctionVariables_device[0]), // input location for basis variables for all steps,
			step,
			maxBasisVariables, 
			numberBasisVariables,
			thrust::raw_pointer_cast(&basisFunctions_workspace_device[0]) // output location
			);
		cudaThreadSynchronize();
	
		if (normalise)
		{
			thrust::copy(basisFunctions_workspace_device.begin(),basisFunctions_workspace_device.begin()+paths,
					basisFunctions_workspace2_device.begin());
			for (int i=1; i < numberDataPoints; ++i)
			{   
				float x = thrust::reduce(basisFunctions_workspace_device.begin()+paths*i,basisFunctions_workspace_device.begin()+paths*(i+1))/paths;
				means_vec[i] = x;
				float y =  sqrt(transform_reduce(basisFunctions_workspace_device.begin()+paths*i,basisFunctions_workspace_device.begin()+paths*(i+1),
					square(), 0.0f, thrust::plus<float>())/paths-x*x);

				sds_vec[i] = y;
				scaleAndShift transformer= { 1/y, - x/y};
				thrust::transform(basisFunctions_workspace_device.begin()+paths*i,basisFunctions_workspace_device.begin()+paths*(i+1),
					basisFunctions_workspace2_device.begin()+paths*i,transformer); 
 
			}
		}
		thrust::transform(continuationValues.begin(),continuationValues.end(), deflatedCashFlowValues_device.begin()+paths*step, continuationValues.begin(),
			thrust::plus<float>());
		cudaThreadSynchronize();
		float alpha=1.0f;
		float beta=0.0f;
		float* regressFunctionsLoc = thrust::raw_pointer_cast(&basisFunctions_workspace_device[0]);
		if (normalise)
			regressFunctionsLoc = thrust::raw_pointer_cast(&basisFunctions_workspace2_device[0]);

			Timer h1;

	reduceOuterProductSymmetric_gpu(paths,
				numberDataPoints,
				threads, 
				blocks, 
				thrust::raw_pointer_cast(&basisFunctions_workspace_device[0]),
				thrust::raw_pointer_cast(&basisExpectation_matrix_device[0]),
				regressFunctionsLoc);
		

	/*

		reduceOuterProductSymmetric_cublas_initted(paths,
			numberDataPoints,
			alpha,
			beta, 
			regressFunctionsLoc,
			thrust::raw_pointer_cast(&basisExpectation_matrix_device[0]));
*/
	/*
		cudaThreadSynchronize();

		reduceOuterProduct_cublas_initted(paths,
			numberDataPoints,
			1,
			alpha, 
			beta, 
			regressFunctionsLoc,
			thrust::raw_pointer_cast(&continuationValues[0]),
			thrust::raw_pointer_cast(&basisproducts_device[0]));

		cudaThreadSynchronize();

		timeInCuBlas += h1.timePassed();
	
	//	thrust::copy(basisExpectation_matrix_device.begin(),basisExpectation_matrix_device.end(),basisExpectation_matrix_host.begin());

		basisExpectation_matrix_host=basisExpectation_matrix_device;
		basisproducts_host=basisproducts_device;
		cudaThreadSynchronize();

#ifdef _DEBUG
		checkVectorForIsNan(basisproducts_host.begin(),basisproducts_host.end(),true);
#endif

	
		MatrixFacade<double> svdInput_mat(&basisExpectation_matrix_host[0],numberDataPoints,numberDataPoints);

		for (size_t i=0; i < basisExpectation_matrix_host.size(); ++i)
			basisExpectation_matrix_host[i] /= paths;

		for (int i=0; i < numberDataPoints; ++i)
			for (int j=0; j < numberDataPoints; ++j)
			{
	//			products_cube(step,i,j) = static_cast<float>(svdInput_mat(i,j));

				if ( useTestMode)
				{
					double x= svdInput_mat(i,j);
					double y= products_gold_cube(step,i,j);
					double z = x-y;
					if ( fabs(z)> testTolerance)
						++numberFailures;

					svdInput_mat(i,j)  = static_cast<float>(products_gold_cube(step,i,j)); // they are within tolerance so make them equal
				}

			}

			for (size_t i=0; i < basisproducts_host.size(); ++i)
			{
				basisproducts_host[i] /= paths;
	//			targets_mat(step,i) = static_cast<float>(basisproducts_host[i]);
				if ( useTestMode)
				{
					double x= basisproducts_host[i];
					double y= targets_gold_mat(step,i);
					if ( fabs(x-y )> testTolerance)
						++numberFailures;

//					targets_mat(step,i) = static_cast<float>(targets_gold_mat(step,i)); // they are within tolerance so make them equal
				}
			}



			Real_d tolerance = 1E-8;
			cudaThreadSynchronize();
	
			svdSolver.SVDSolve<double>(svdInput_mat, 
				&basisproducts_host[0], 
				&coefficients_step_vec[0],
 			tolerance);

			if (normalise)
			{
				
				for (int i=1; i < numberDataPoints; ++i)
				{
					coefficients_step_vec[0] -= coefficients_step_vec[i]*means_vec[i]/sds_vec[i];
				}

				for (int i=1; i < numberDataPoints; ++i)
				{
					coefficients_step_vec[i] /= sds_vec[i];
				}

			}


			for (size_t i=0; i < coefficients_step_vec.size(); ++i)
			{
				regressCoeffs_mat(step,i) = coefficients_step_vec[i];
				regressCoeffs_host_mat(step,i) = static_cast<float>(coefficients_step_vec[i]);

				if (useTestMode)
				{
					double x= regressCoeffs_host_mat(step,i);
					double y= coefficients_gold(step,i);
					if ( fabs(x-y )> testTolerance)
						++numberFailures;

					coefficients_step_vec[i] = static_cast<float>(coefficients_gold(step,i));

					regressCoeffs_host_mat(step,i)=static_cast<float>(coefficients_gold(step,i));


				}
			}

			// we now have the regression coeffs on the CPU
			// we need to get then onto the GPU

			thrust::copy(coefficients_host.begin()+step*numberDataPoints,
				coefficients_host.begin()+(step+1)*numberDataPoints,
				coefficients_device.begin()+step*numberDataPoints);

			float andersenShift =0.0f;

	//			debugDumpVector(continuationValues,"continuationValues pre");

	
			oneStepUpdateLS_main( paths,
				step,
				basisFunctions_workspace_device, // just the function values for this step, unnormalised 
				numberDataPoints,
				coefficients_device, // the weights for all steps
				continuationValues,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
				deflatedCashFlowValues_device,  //the cash flow values for all steps deflated to exercise times
				deflatedExerciseValues_device,      // the exerciseValues for all steps deflated to exercise times
				andersenShift);
	//	debugDumpVector(continuationValues,"continuationValues post");


			if (step >0)
				if (deflation_locations_vec[step-1]!=deflation_locations_vec[step]  ) 
				updateDeflationOfContinuationValues_main( continuationValues,  
				numeraireValues_device,
				paths,
				deflation_locations_vec[step-1],
				deflation_locations_vec[step]);
	

	}


	double res = thrust::inner_product(continuationValues.begin(),continuationValues.end(),numeraireValues_device.begin(), 0.0f);

	res/=paths;
	cublasShutdown();

	std::copy(coefficients_host.begin(),coefficients_host.end(), coefficients_vec.begin());

	testPassed = numberFailures ==0;

	std::cout << " time in cublas " << timeInCuBlas << "\n";

	return res;
}

template<class T>
double generateRegressionCoefficientsViaLS_gpu(int numberExerciseDates,
			//								   std::vector<float>& products_cube_vec,
			//								   std::vector<float>& targets_mat_vec,
											   thrust::device_vector<int>& basisVariableInt_data_device,
											   thrust::device_vector<float>& basisVariableFloat_data_device,
											   thrust::device_vector<float>& coefficients_device, // the LS coefficients are placed here 
											   std::vector<Realv>& coefficients_vec, // the LS coefficients are also placed here 
											   thrust::device_vector<float>&  basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
											   thrust::device_vector<int>& basisVariablesEachStep_device, //vector of the number of basis variables for each step
											   const std::vector<int>& basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
											   int maxBasisVariables,
											   thrust::device_vector<float>& deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
											   thrust::device_vector<float>& deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
											   thrust::device_vector<float>& numeraireValues_device, // numeraire vals for each, matrix of times and paths 
											   const std::vector<int>&	deflation_locations_vec,
											   int paths,
											   int threads,
											   int blocks,
											   bool normalise								  
											   )
{
	 bool useTestMode = false;

	 bool testPassed = false;
	 std::vector<double> v(1);
	 CubeConstFacade<double> products_gold_cube(v,1,1,1); // ignored 
	 MatrixConstFacade<double> targets_gold_mat(v,1,1); // ignored 
	 MatrixConstFacade<double> coefficients_gold(v,1,1);                                 // ignored if not in testmode
	 double testTolerance =0.0;

	return  generateRegressionCoefficientsViaLS_gpu<T>(numberExerciseDates,
											    basisVariableInt_data_device,
											    basisVariableFloat_data_device,
											    coefficients_device, // the LS coefficients are placed here 
											   coefficients_vec, // the LS coefficients are also placed here 
											     basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
											   basisVariablesEachStep_device, //vector of the number of basis variables for each step
											   basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
											   maxBasisVariables,
											   deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
											   deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
											    numeraireValues_device, // numeraire vals for each, matrix of times and paths 
												deflation_locations_vec,
											   paths,
											   threads,
											   blocks,
											  normalise,
											  useTestMode,
											  testPassed,
											   products_gold_cube, // ignored if not in testmode
											  targets_gold_mat,// ignored if not in testmode
											   coefficients_gold,                                 // ignored if not in testmode
											   testTolerance   // ignored if not in testmode										  
											   );
}
*/
#endif

