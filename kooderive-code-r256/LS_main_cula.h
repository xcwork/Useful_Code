//
//
//                  LS_main_cula.h
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef LS_MAIN_CULA_H
#define LS_MAIN_CULA_H
#include <LS_main.h>
#include <RegressionSelectorBase.h>
#include <thrust/device_ptr.h>

#include <Utilities.h>
#include <gold/Select_gold.h>
#include <ComparisonCheck.h>
#include <string>
#include <Matrix_solve_main.h>

#include <LinearSolver_base_main.h>
/*
double generateRegressionCoefficientsViaLSquadratic_cula_gpu(int numberExerciseDates,
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

/*
template<class T>
double generateRegressionCoefficientsViaLS_gpu_cula(int numberExerciseDates,
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
													int paths)													
{
	double res = -1.0;
	{
	culaInitialize();


	int numberDataPoints= T::functionValues(maxBasisVariables);

	
	thrust::device_vector<float> basisFunctions_workspace_device(numberDataPoints  * paths );
	
	
	thrust::device_vector<float> basisFunctions_workspace2_device(numberDataPoints  * paths );
	
	
	thrust::device_vector<float> basisExpectation_matrix_device(numberDataPoints  * numberDataPoints );
	
	
	thrust::device_vector<float> basisproducts_device(numberDataPoints   );
	
	
	thrust::host_vector<double> basisExpectation_matrix_host(basisExpectation_matrix_device.size());
	
	
	thrust::host_vector<double> basisproducts_host(basisproducts_device.size());
	
	
	thrust::device_vector<float> cvals_workspace(paths);
	

	thrust::device_vector<float> continuationValues(paths,0.0f);
	
	int oldnumberBasisVariables = *basisVariablesEachStep_vec.rbegin();
	
	

	SVDSquareMatrix svdSolver(T::functionValues(*basisVariablesEachStep_vec.rbegin()));

	coefficients_vec.resize(numberDataPoints*numberExerciseDates);
	thrust::host_vector<float> coefficients_host(coefficients_vec.size());
	std::vector<double> coefficients_step_vec(numberDataPoints);

	MatrixFacade<double> regressCoeffs_mat(&coefficients_vec[0],numberExerciseDates,numberDataPoints);
	MatrixFacade<float> regressCoeffs_host_mat(&coefficients_host[0],numberExerciseDates,numberDataPoints);

	int maxNumberDataPoints = numberDataPoints;
	

	for (int step= numberExerciseDates-1;  step >=0;--step)
	{
	
		int numberBasisVariables = basisVariablesEachStep_vec[step];
		int numberDataPoints = T::functionValues(numberBasisVariables);


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

		thrust::transform(continuationValues.begin(),continuationValues.end(), deflatedCashFlowValues_device.begin()+paths*step, continuationValues.begin(),
			thrust::plus<float>());
		cudaThreadSynchronize();

		float* regressFunctionsLoc = thrust::raw_pointer_cast(&basisFunctions_workspace_device[0]);

		Timer h1;
#ifdef _DEBUG
		checkVectorForIsNan(basisproducts_host.begin(),basisproducts_host.end(),true);
#endif

		//-- location of A matrix
		basisFunctions_workspace2_device = basisFunctions_workspace_device;
		float* A_mat_global = thrust::raw_pointer_cast(&basisFunctions_workspace2_device[0]);
		// location of target vector

		//	float* b = thrust::raw_pointer_cast(&coefficients_device[step*maxNumberDataPoints]);

		cvals_workspace = continuationValues;

		float* cont_val_wsp_global = thrust::raw_pointer_cast(&cvals_workspace[0]);		

		// culaSolve works in place so we have to copy the data beforehand
		//culaSolve(paths,numberDataPoints, A_mat_global,cont_val_wsp_global,'T');
		culaSolve(numberDataPoints,paths, A_mat_global,cont_val_wsp_global,'N');

		thrust::copy(cvals_workspace.begin(),cvals_workspace.begin()+numberDataPoints, coefficients_device.begin()+step*maxNumberDataPoints);

//		debugDumpVector(cvals_workspace,"cvals_workspace");
//		debugDumpVector(continuationValues,"continuationValues");
//
//		debugDumpVector(coefficients_device,"coefficients_device");
//		debugDumpMatrix(basisFunctions_workspace_device,"basisFunctions_workspace_device", maxNumberDataPoints,paths);

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
	
	std::cout << " about to inner product...\n";
	std::cout << continuationValues.size()<< " , " << numeraireValues_device.size() << "\n";
	res = doInnerProduct(continuationValues, numeraireValues_device);

	
	
	std::cout << " about to assign..\n";
	coefficients_host= coefficients_device;
	
	std::cout << " about to copy..\n";
	std::copy(coefficients_host.begin(),coefficients_host.end(),coefficients_vec.begin());

	std::cout << " about to destroy...\n";

	}
	res/=paths;
	culaShutdown();

		return res;
}
*/
    /*
double generateRegressionCoefficientsViaLSMultiquadratic_cula_main(int numberExerciseDates,
																   //								   std::vector<float>& products_cube_vec,
																   //								   std::vector<float>& targets_mat_vec,
																   thrust::device_vector<int>& basisVariableInt_data_device,
																   thrust::device_vector<float>& basisVariableFloat_data_device,
																   thrust::device_vector<float>& coefficients_device, // the LS coefficients are placed here 
																   std::vector<Realv>& coefficients_vec, // the LS coefficients are also placed here 
																   thrust::device_vector<float>& lowercuts_device, // the lower cut points for the regressions 
																   thrust::device_vector<float>& uppercuts_device, // the upper cut points for the regressions 
																   thrust::device_vector<float>&  basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
																   thrust::device_vector<int>& basisVariablesEachStep_device, //vector of the number of basis variables for each step
																   const std::vector<int>& basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
																   int maxBasisVariables,
																   thrust::device_vector<float>& deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
																   thrust::device_vector<float>& deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
																   thrust::device_vector<float>& numeraireValues_device, // numeraire vals for each, matrix of times and paths 
															//	   const std::vector<int>& deflation_locations_vec,
																   int paths,
																   bool useCrossTerms,
																   int numberOfExtraRegressions,
																   RegressionSelector& selector,
																   int minPathsLeft);

                                                                   */

double generateRegressionCoefficientsViaLSMultiquadratic_flexi_main(int numberExerciseDates,
														 //								   std::vector<float>& products_cube_vec,
														 //								   std::vector<float>& targets_mat_vec,
														 thrust::device_vector<int>& basisVariableInt_data_device,
														 thrust::device_vector<float>& basisVariableFloat_data_device,
														 thrust::device_vector<float>& coefficients_device, // the LS coefficients are placed here 
														 std::vector<Realv>& coefficients_vec, // the LS coefficients are also placed here 
														 thrust::device_vector<float>& lowercuts_device, // the lower cut points for the regressions 
														 thrust::device_vector<float>& uppercuts_device, // the upper cut points for the regressions 
														 thrust::device_vector<float>&  basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
														 thrust::device_vector<int>& basisVariablesEachStep_device, //vector of the number of basis variables for each step
														 const std::vector<int>& basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
														 int maxBasisVariables,
														 thrust::device_vector<float>& deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
														 thrust::device_vector<float>& deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
														 thrust::device_vector<float>& numeraireValues_device, // numeraire vals for each, matrix of times and paths 
										//				 const std::vector<int>& deflation_locations_vec,
														 int paths,
														 bool useCrossTerms,
														 int numberOfExtraRegressions,
														 RegressionSelector& selector,
														 int minPathsLeft,
														 LinearSolverBaseMain& solver);

void debugDumpAATAndAx(thrust::device_vector<float>& A,thrust::device_vector<float>& target, int rows, int columns,std::string identifier);

/*
template<class T>
double generateMultiRegressionCoefficientsViaLS_gpu_cula(int numberExerciseDates,
														 //								   std::vector<float>& products_cube_vec,
														 //								   std::vector<float>& targets_mat_vec,
														 thrust::device_vector<int>& basisVariableInt_data_device,
														 thrust::device_vector<float>& basisVariableFloat_data_device,
														 thrust::device_vector<float>& coefficients_device, // the LS coefficients are placed here 
														 std::vector<Realv>& coefficients_vec, // the LS coefficients are also placed here 
														 thrust::device_vector<float>& lowercuts_device, // the lower cut points for the regressions 
														 thrust::device_vector<float>& uppercuts_device, // the upper cut points for the regressions 
														 thrust::device_vector<float>&  basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
														 thrust::device_vector<int>& basisVariablesEachStep_device, //vector of the number of basis variables for each step
														 const std::vector<int>& basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
														 int maxBasisVariables,
														 thrust::device_vector<float>& deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
														 thrust::device_vector<float>& deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
														 thrust::device_vector<float>& numeraireValues_device, // numeraire vals for each, matrix of times and paths 
									//					 const std::vector<int>& deflation_locations_vec,
														 int paths,
														 int numberOfExtraRegressions,
														 RegressionSelector& selector,
														 int minPathsForRegression)													
{
	culaInitialize();



	int numberDataPoints= T::functionValues(maxBasisVariables);
	int maxNumberDataPoints = numberDataPoints;
	thrust::device_vector<float> basisFunctions_workspace_device(numberDataPoints  * paths );
	thrust::device_vector<float> basisFunctions_workspace2_device(numberDataPoints  * paths );
	thrust::device_vector<float> basisExpectation_matrix_device(numberDataPoints  * numberDataPoints );
	thrust::device_vector<float> basisproducts_device(numberDataPoints   );

	thrust::host_vector<double> basisExpectation_matrix_host(basisExpectation_matrix_device.size());
	thrust::host_vector<double> basisproducts_host(basisproducts_device.size());

	thrust::device_vector<float> cvals_workspace(paths);
	thrust::device_vector<float> cvals_workspace3_device(paths+1);

	thrust::device_vector<float> continuationValues(paths,0.0f);

	int oldnumberBasisVariables = *basisVariablesEachStep_vec.rbegin();


	SVDSquareMatrix svdSolver(T::functionValues(*basisVariablesEachStep_vec.rbegin()));

	coefficients_vec.resize(numberDataPoints*numberExerciseDates);
	thrust::host_vector<float> coefficients_host(coefficients_vec.size());
	std::vector<double> coefficients_step_vec(numberDataPoints);

	MatrixFacade<double> regressCoeffs_mat(&coefficients_vec[0],numberExerciseDates,numberDataPoints);
	MatrixFacade<float> regressCoeffs_host_mat(&coefficients_host[0],numberExerciseDates,numberDataPoints);

	//extra data for multiregression

	thrust::device_vector<float> basisFunctions_workspace3_device(numberDataPoints  * paths +1);
	thrust::device_vector<float> cvals_workspace2_device(paths+1);

	thrust::device_vector<float> regressed_path_values_device(paths);
	thrust::device_vector<int> selected_device(paths);
	thrust::device_vector<int> selected_index_device(paths);

	// layer is step, which regression is row, column is basis function index
	int coefficientsLayerSize = maxNumberDataPoints*(numberOfExtraRegressions+1);
	coefficients_device.resize(numberExerciseDates*coefficientsLayerSize);
	coefficients_vec.resize(coefficients_device.size());

	thrust::device_vector<float> estCval_device(paths);
	thrust::device_vector<int> exercise_device(paths);



	for (int step= numberExerciseDates-1;  step >=0;--step)
	{
		int numberBasisVariables = basisVariablesEachStep_vec[step];
		int numberDataPoints = T::functionValues(numberBasisVariables);


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

		thrust::transform(continuationValues.begin(),continuationValues.end(), deflatedCashFlowValues_device.begin()+paths*step, continuationValues.begin(),
			thrust::plus<float>());
		cudaThreadSynchronize();

		float* regressFunctionsLoc = thrust::raw_pointer_cast(&basisFunctions_workspace_device[0]);

		Timer h1;
#ifdef _DEBUG
		checkVectorForIsNan(basisproducts_host.begin(),basisproducts_host.end(),true);
#endif

		//-- location of A matrix
		basisFunctions_workspace2_device = basisFunctions_workspace_device;
		float* A_mat_global = thrust::raw_pointer_cast(&basisFunctions_workspace2_device[0]);
		// location of target vector

		//	float* b = thrust::raw_pointer_cast(&coefficients_device[step*maxNumberDataPoints]);

		cvals_workspace = continuationValues;

		float* cont_val_wsp_global = thrust::raw_pointer_cast(&cvals_workspace[0]);		

		culaSolve(numberDataPoints,paths, A_mat_global,cont_val_wsp_global,'N');

		thrust::copy(cvals_workspace.begin(),cvals_workspace.begin()+numberDataPoints, coefficients_device.begin()+step*coefficientsLayerSize);

		

//        debugDumpAATAndAx(basisFunctions_workspace_device,continuationValues, numberDataPoints, paths,"basisFunctions_workspace_device");
	//		debugDumpMatrix(basisFunctions_workspace_device,"basisFunctions_workspace_device",numberDataPoints,paths);
		//		debugDumpVector(continuationValues,"continuationValues");


		//	std::vector<float> tmp(stlVecFromDevVec(coefficients_device));

		float andersenShift =0.0f;

		int pathsLeft = paths;

		thrust::device_vector<float> remainingbasisFunctions_device(basisFunctions_workspace_device);
		thrust::device_vector<float> remainingContinuationValues_device(continuationValues);
		thrust::device_vector<float> remainingExerciseValues_device(deflatedExerciseValues_device.begin()+step*paths,deflatedExerciseValues_device.begin()+(step+1)*paths);
		thrust::device_vector<float> remainingRegressedContinuationValues_device(continuationValues);


		bool sufficientPathsLeft = true;

		for (int i=0; i < numberOfExtraRegressions && sufficientPathsLeft; ++i)
		{
			basisFunctions_workspace3_device.resize(pathsLeft*maxNumberDataPoints+1);


			// evaluate net continuation values
			double t= est_continuation_net_value_evaluation_main( pathsLeft,
				step,
				maxNumberDataPoints,
				numberOfExtraRegressions+1,
				i,
				remainingbasisFunctions_device, // just the function values for this step
				numberDataPoints,
				coefficients_device, // the weights for all steps
				remainingExerciseValues_device,      // the exerciseValues for this steps deflated 
				andersenShift,
				remainingRegressedContinuationValues_device);

			// select paths

			float lowerCutOff,upperCutOff;

			thrust::device_vector<float>::iterator  start =
				remainingRegressedContinuationValues_device.begin();
			thrust::device_vector<float>::iterator  end = start+pathsLeft;
			thrust::device_vector<int>::iterator  selectstart= selected_device.begin();

			//	bool testPassed = false;

			int pathsAfter =  selector.Select(i,
				start,
				end,
				selectstart,
				lowerCutOff,
				upperCutOff//, testPassed
				);

			if (pathsAfter > minPathsForRegression)
			{
				lowercuts_device[step*(numberOfExtraRegressions+1)+i] = lowerCutOff;
				uppercuts_device[step*(numberOfExtraRegressions+1)+i] = upperCutOff;


	//			std::cout << step << "," << i << "," << pathsAfter << "," << lowerCutOff << "," << upperCutOff << "\n";

				// grab paths



				PartialSumsInt(selected_device,pathsLeft,selected_index_device);



				doScatter(remainingExerciseValues_device,0, pathsLeft,pathsAfter, selected_index_device,
					selected_device, 
					cvals_workspace2_device,0
					);



				doScatter(remainingContinuationValues_device,0, pathsLeft,pathsAfter, selected_index_device,
					selected_device, 
					cvals_workspace2_device,0
					);



				doScatterMulti(remainingbasisFunctions_device,numberDataPoints, pathsLeft,pathsAfter, selected_index_device,
					selected_device, 
					basisFunctions_workspace3_device
					);


				// we now have the selected paths in remaining... for first pathsAfter elements

				pathsLeft = pathsAfter;

		//		std::cout << "paths left, " << pathsLeft << "\n";

				//		float* regressFunctionsLoc = thrust::raw_pointer_cast(&basisFunctions_workspace_device[0]);
				
    //            debugDumpAATAndAx(remainingbasisFunctions_device,remainingContinuationValues_device,numberDataPoints , pathsAfter,"remainingbasisFunctions_device");
	//			debugDumpMatrix(remainingbasisFunctions_device,"remainingbasisFunctions_device",numberDataPoints,pathsAfter);
	//			debugDumpVector(remainingContinuationValues_device,"remainingContinuationValues_device");


				basisFunctions_workspace3_device = remainingbasisFunctions_device;
				//-- location of A matrix
				float* A_mat_global = thrust::raw_pointer_cast(&basisFunctions_workspace3_device[0]);
				// location of target vector

				//	float* b = thrust::raw_pointer_cast(&coefficients_device[step*maxNumberDataPoints]);

				cvals_workspace = remainingContinuationValues_device;

				float* cont_val_wsp_global = thrust::raw_pointer_cast(&cvals_workspace[0]);		

				culaSolve(numberDataPoints,pathsLeft, A_mat_global,cont_val_wsp_global,'N');

				thrust::copy(cvals_workspace.begin(),
					cvals_workspace.begin()+numberDataPoints,
					coefficients_device.begin()+step*coefficientsLayerSize+(i+1)*maxNumberDataPoints);
			}
			else
			{
				lowercuts_device[step*(numberOfExtraRegressions+1)+i] = 0.0f;
				uppercuts_device[step*(numberOfExtraRegressions+1)+i] = 0.0f;;

				sufficientPathsLeft=false;


			}


		}

	//	debugDumpVector(continuationValues,"continuationValues pre");


		oneStepUpdateLS_multi_main( paths,
			step,
			basisFunctions_workspace_device, // just the function values for this step, unnormalised 
			numberDataPoints,
			maxNumberDataPoints,
			numberOfExtraRegressions+1,
			coefficients_device, // the weights for all steps
			lowercuts_device,
			uppercuts_device,
			continuationValues,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
			deflatedCashFlowValues_device,  //the cash flow values for all steps deflated to exercise times
			deflatedExerciseValues_device,      // the exerciseValues for all steps deflated to exercise times
			andersenShift,
			estCval_device,
			exercise_device);
			
	//	debugDumpVector(continuationValues,"continuationValues post");
//		debugDumpVector(estCval_device,"estCval_device ");
	//	debugDumpVector(exercise_device,"exercise_device ");

		if (step >0)
				updateDeflationOfContinuationValues_main( continuationValues,  
				numeraireValues_device,
				paths,
				step-1,
				step);


	}


	double res = thrust::inner_product(continuationValues.begin(),continuationValues.end(),numeraireValues_device.begin(), 0.0f,thrust::plus<float>(),thrust::divides<float>());

//	debugDumpVector(continuationValues,"continuationValues at end");


	res/=paths;
	culaShutdown();

	coefficients_host= coefficients_device;
	std::copy(coefficients_host.begin(),coefficients_host.end(),coefficients_vec.begin());
	return res;
}
*/

template<class T>
double generateMultiRegressionCoefficientsViaLS_gpu_flexisolve(int numberExerciseDates,
														 //								   std::vector<float>& products_cube_vec,
														 //								   std::vector<float>& targets_mat_vec,
														 thrust::device_vector<int>& basisVariableInt_data_device,
														 thrust::device_vector<float>& basisVariableFloat_data_device,
														 thrust::device_vector<float>& coefficients_device, // the LS coefficients are placed here 
														 std::vector<Realv>& coefficients_vec, // the LS coefficients are also placed here 
														 thrust::device_vector<float>& lowercuts_device, // the lower cut points for the regressions 
														 thrust::device_vector<float>& uppercuts_device, // the upper cut points for the regressions 
														 thrust::device_vector<float>&  basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
														 thrust::device_vector<int>& basisVariablesEachStep_device, //vector of the number of basis variables for each step
														 const std::vector<int>& basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
														 int maxBasisVariables,
														 thrust::device_vector<float>& deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
														 thrust::device_vector<float>& deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
														 thrust::device_vector<float>& numeraireValues_device, // numeraire vals for each, matrix of times and paths 
									//					 const std::vector<int>& deflation_locations_vec,
														 int paths,
														 int numberOfExtraRegressions,
														 RegressionSelector& selector,
														 int minPathsForRegression,
														 LinearSolverBaseMain& solver)													
{
	double timeDataMoving =0.0;
	double timeUpdating =0.0;
	double timeSpentExpanding =0.0;

	double timeOnSetup=0.0;
	double timeOnCopying =0.0;
	double timeSpentCollating =0.0;

	Timer setupTimer;

	int numberDataPoints= T::functionValues(maxBasisVariables);
	int maxNumberDataPoints = numberDataPoints;
	thrust::device_vector<float> basisFunctions_workspace_device(numberDataPoints  * paths );
	thrust::device_vector<float> basisFunctions_workspace2_device(numberDataPoints  * paths );
	thrust::device_vector<float> basisExpectation_matrix_device(numberDataPoints  * numberDataPoints );
	thrust::device_vector<float> basisproducts_device(numberDataPoints   );

	thrust::host_vector<double> basisExpectation_matrix_host(basisExpectation_matrix_device.size());
	thrust::host_vector<double> basisproducts_host(basisproducts_device.size());

	thrust::device_vector<float> cvals_workspace(paths);
	thrust::device_vector<float> cvals_workspace3_device(paths+1);

	thrust::device_vector<float> continuationValues(paths,0.0f);

	int oldnumberBasisVariables = *basisVariablesEachStep_vec.rbegin();

	coefficients_vec.resize(numberDataPoints*numberExerciseDates);
	thrust::host_vector<float> coefficients_host(coefficients_vec.size());
	std::vector<double> coefficients_step_vec(numberDataPoints);

	MatrixFacade<double> regressCoeffs_mat(&coefficients_vec[0],numberExerciseDates,numberDataPoints);
	MatrixFacade<float> regressCoeffs_host_mat(&coefficients_host[0],numberExerciseDates,numberDataPoints);

	//extra data for multiregression

	thrust::device_vector<float> basisFunctions_workspace3_device(numberDataPoints  * paths +1);
	thrust::device_vector<float> cvals_workspace2_device(paths+1);

	thrust::device_vector<float> regressed_path_values_device(paths);
	thrust::device_vector<int> selected_device(paths);
	thrust::device_vector<int> selected_index_device(paths);

	// layer is step, which regression is row, column is basis function index
	int coefficientsLayerSize = maxNumberDataPoints*(numberOfExtraRegressions+1);
	coefficients_device.resize(numberExerciseDates*coefficientsLayerSize);
	coefficients_vec.resize(coefficients_device.size());

	thrust::device_vector<float> estCval_device(paths);
	thrust::device_vector<int> exercise_device(paths);

		thrust::device_vector<float> remainingbasisFunctions_device(basisFunctions_workspace_device.size());
		thrust::device_vector<float> remainingContinuationValues_device(continuationValues.size());
		thrust::device_vector<float> remainingExerciseValues_device(paths);
		thrust::device_vector<float> remainingRegressedContinuationValues_device(continuationValues.size());

	timeOnSetup = setupTimer.timePassed();

	for (int step= numberExerciseDates-1;  step >=0;--step)
	{
		int numberBasisVariables = basisVariablesEachStep_vec[step];
		int numberDataPoints = T::functionValues(numberBasisVariables);

		Timer expansionTimer;

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

		timeSpentExpanding += expansionTimer.timePassed();


		Timer collationTimer;

		thrust::transform(continuationValues.begin(),continuationValues.end(), deflatedCashFlowValues_device.begin()+paths*step, continuationValues.begin(),
			thrust::plus<float>());
		cudaThreadSynchronize();

	    timeSpentCollating += collationTimer.timePassed();

		float* regressFunctionsLoc = thrust::raw_pointer_cast(&basisFunctions_workspace_device[0]);

		Timer h1;
#ifdef _DEBUG
		checkVectorForIsNan(basisproducts_host.begin(),basisproducts_host.end(),true);
#endif

		solver.setDimensions(paths, numberDataPoints);
	    solver.solveSystem(basisFunctions_workspace_device, continuationValues, cvals_workspace  );

		thrust::copy(cvals_workspace.begin(),cvals_workspace.begin()+numberDataPoints, coefficients_device.begin()+step*coefficientsLayerSize);

		
		float andersenShift =0.0f;

		int pathsLeft = paths;
	
		Timer copyTimer;

		remainingbasisFunctions_device = basisFunctions_workspace_device;
		remainingContinuationValues_device=continuationValues;
		thrust::copy(remainingExerciseValues_device.begin(),remainingExerciseValues_device.end(),deflatedExerciseValues_device.begin()+step*paths);
		remainingRegressedContinuationValues_device=continuationValues;
		
		cudaThreadSynchronize();
		timeOnCopying += copyTimer.timePassed();

		bool sufficientPathsLeft = true;

		for (int i=0; i < numberOfExtraRegressions && sufficientPathsLeft; ++i)
		{
			basisFunctions_workspace3_device.resize(pathsLeft*maxNumberDataPoints+1);


			// evaluate net continuation values
			double t= est_continuation_net_value_evaluation_main( pathsLeft,
				step,
				maxNumberDataPoints,
				numberOfExtraRegressions+1,
				i,
				remainingbasisFunctions_device, // just the function values for this step
				numberDataPoints,
				coefficients_device, // the weights for all steps
				remainingExerciseValues_device,      // the exerciseValues for this steps deflated 
				andersenShift,
				remainingRegressedContinuationValues_device);

			// select paths

			float lowerCutOff,upperCutOff;

			thrust::device_vector<float>::iterator  start =
				remainingRegressedContinuationValues_device.begin();
			thrust::device_vector<float>::iterator  end = start+pathsLeft;
			thrust::device_vector<int>::iterator  selectstart= selected_device.begin();

	
			int pathsAfter =  selector.Select(i,start,end,selectstart,lowerCutOff,upperCutOff);

			if (pathsAfter > minPathsForRegression)
			{
				lowercuts_device[step*(numberOfExtraRegressions+1)+i] = lowerCutOff;
				uppercuts_device[step*(numberOfExtraRegressions+1)+i] = upperCutOff;

				// grab paths

				Timer dataMoveTimer;

				PartialSumsInt(selected_device,pathsLeft,selected_index_device);

				doScatter(remainingExerciseValues_device,0, pathsLeft,pathsAfter, selected_index_device,
					selected_device, 
					cvals_workspace2_device,0
					);

				doScatter(remainingContinuationValues_device,0, pathsLeft,pathsAfter, selected_index_device,
					selected_device, 
					cvals_workspace2_device,0
					);

				doScatterMulti(remainingbasisFunctions_device,numberDataPoints, pathsLeft,pathsAfter, selected_index_device,
					selected_device, 
					basisFunctions_workspace3_device
					);

				cudaThreadSynchronize();

				timeDataMoving+=dataMoveTimer.timePassed();
				// we now have the selected paths in remaining... for first pathsAfter elements

				pathsLeft = pathsAfter;

				solver.setDimensions(pathsLeft, numberDataPoints);
	        	solver.solveSystem(remainingbasisFunctions_device, remainingContinuationValues_device, cvals_workspace  );

				thrust::copy(cvals_workspace.begin(),
					cvals_workspace.begin()+numberDataPoints,
					coefficients_device.begin()+step*coefficientsLayerSize+(i+1)*maxNumberDataPoints);
			}
			else
			{
				lowercuts_device[step*(numberOfExtraRegressions+1)+i] = 0.0f;
				uppercuts_device[step*(numberOfExtraRegressions+1)+i] = 0.0f;;

				sufficientPathsLeft=false;
			}
		}

		Timer updatingTimer;

		oneStepUpdateLS_multi_main( paths,
			step,
			basisFunctions_workspace_device, // just the function values for this step, unnormalised 
			numberDataPoints,
			maxNumberDataPoints,
			numberOfExtraRegressions+1,
			coefficients_device, // the weights for all steps
			lowercuts_device,
			uppercuts_device,
			continuationValues,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
			deflatedCashFlowValues_device,  //the cash flow values for all steps deflated to exercise times
			deflatedExerciseValues_device,      // the exerciseValues for all steps deflated to exercise times
			andersenShift,
			estCval_device,
			exercise_device);
			
	//	debugDumpVector(continuationValues,"continuationValues post");
//		debugDumpVector(estCval_device,"estCval_device ");
	//	debugDumpVector(exercise_device,"exercise_device ");

		if (step >0)
				updateDeflationOfContinuationValues_main( continuationValues,  
				numeraireValues_device,
				paths,
				step-1,
				step);

		cudaThreadSynchronize();

		timeUpdating += updatingTimer.timePassed();
	}


	double res = thrust::inner_product(continuationValues.begin(),continuationValues.end(),numeraireValues_device.begin(), 0.0f,thrust::plus<float>(),thrust::divides<float>());

//	debugDumpVector(continuationValues,"continuationValues at end");


	res/=paths;
//	culaShutdown();

	std::cout << " time spent data moving " << timeDataMoving << "\n";

	std::cout << " time spent data updating " << timeUpdating << "\n";
    std::cout << " time spent data expanding " << timeSpentExpanding << "\n";
    std::cout << " time on set up " << timeOnSetup << "\n";
    std::cout << " time on copying " << timeOnCopying<< "\n";

   std::cout << " time on collating " << timeSpentCollating<< "\n";
	coefficients_host= coefficients_device;
	std::copy(coefficients_host.begin(),coefficients_host.end(),coefficients_vec.begin());
	return res;
}
#endif
