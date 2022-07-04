//
//
//                  LS_main.cu
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#include <LS_main.h>
#include <LS_gpu.h>

void expandBasisFunctions_quadratic_main(
	int totalNumberOfPaths, 
	thrust::device_vector<float>& basisVariables_device, // input location for basis variables for all steps,
	int exerciseNumber,
	int maxBasisVariableSize, 
	int variableSizeThisStep,
	bool useCrossTerms,
	thrust::device_vector<float>& basisFunctions_device
	)
{

	expandBasisFunctions_quadratic_gpu(
		totalNumberOfPaths, 
		thrust::raw_pointer_cast(&basisVariables_device[0]), // input location for basis variables for all steps,
		exerciseNumber,
		maxBasisVariableSize, 
		variableSizeThisStep,
		useCrossTerms,
		thrust::raw_pointer_cast(&basisFunctions_device[0])
		);

}

double est_continuation_net_value_evaluation_main(int pathsRemaining,
												  int exerciseNumber,
												  int maxNumberBasisFunctions,
												  int maxNumberOfRegressions,
												  int whichRegression,
												  thrust::device_vector<float>& basisFunctionsRemaining_device, // just the function values for this step
												  int numberBasisFunctions,
												  thrust::device_vector<float>& basis_weights_device, // the weights for all steps all regressions
												  thrust::device_vector<float>& exerciseValues_device,      // the exerciseValues for this steps deflated to  exercise times
												  float AndersenShift,
												  thrust::device_vector<float>& output_device)
{
	return continuation_net_value_evaluation_gpu(pathsRemaining,
		exerciseNumber,
		thrust::raw_pointer_cast(&basisFunctionsRemaining_device[0]),
		numberBasisFunctions,
		thrust::raw_pointer_cast(&basis_weights_device[exerciseNumber*maxNumberBasisFunctions*maxNumberOfRegressions+whichRegression*maxNumberBasisFunctions]), // the weights for all steps
		thrust::raw_pointer_cast(&exerciseValues_device[0]),      // the exerciseValues for this steps deflated to  exercise times
		AndersenShift,
		thrust::raw_pointer_cast(&output_device[0]));
}

/*

double oneStepUpdateLS_main(int paths,
							int exerciseNumber,
							thrust::device_vector<float>& basisFunctions_device, // just the function values for this step
							int numberBasisFunctions,
							thrust::device_vector<float>& basis_weights_device, // the weights for all steps
							thrust::device_vector<float>& continuationValues_device,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
							thrust::device_vector<float>& deflatedCashValues_device,  //the cash flow values for all steps deflated to exercise times
							thrust::device_vector<float>& exerciseValues_device,      // the exerciseValues for all steps deflated to exercise times
							float AndersenShift)
{



	return oneStepUpdateLS_gpu( paths,
		exerciseNumber,
		thrust::raw_pointer_cast(&basisFunctions_device[0]), // just the function values for this step
		numberBasisFunctions,
		thrust::raw_pointer_cast(&basis_weights_device[0]), // the weights for all steps
		thrust::raw_pointer_cast(&continuationValues_device[0]),  //the continuation values at the moment, this will be overwritten, deflated to current ex time
		thrust::raw_pointer_cast(&deflatedCashValues_device[0]),  //the continuation values at the moment, this will be overwritten, deflated to current ex time
		thrust::raw_pointer_cast(&exerciseValues_device[0]),      // the exerciseValues for this step deflated to current exercise time
		AndersenShift);

}
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
								   thrust::device_vector<int>& exercises_device
								   )
{

	

	return oneStepUpdateLS_multi_gpu( paths,
		exerciseNumber,
		thrust::raw_pointer_cast(&basisFunctions_device[0]), // just the function values for this step
		numberBasisFunctions,
		basisFunctionsRowSize,
		maxRegressionDepth,
		thrust::raw_pointer_cast(&basis_weights_device[0]), // the weights for all steps
		thrust::raw_pointer_cast(&lowerCuts_device[exerciseNumber*maxRegressionDepth]),
		thrust::raw_pointer_cast(&upperCuts_device[exerciseNumber*maxRegressionDepth]),
		thrust::raw_pointer_cast(&continuationValues_device[0]),  //the continuation values at the moment, this will be overwritten, deflated to current ex time
		thrust::raw_pointer_cast(&deflatedCashFlows_device[0]),   // the cash-flow values all steps deflated to ex times
		thrust::raw_pointer_cast(&exerciseValues_device[0]),      // the exerciseValues for all steps deflated to  exercise times
		AndersenShift,
		thrust::raw_pointer_cast(&estContValues_device[0]),
		thrust::raw_pointer_cast(&exercises_device[0]));

}

double updateDeflationOfContinuationValues_main(  thrust::device_vector<float>& deflatedNextStepValues_device,  
												thrust::device_vector<float>& numeraireValues_device,
												int paths,
												int newStepNumber,
												int oldStepNumber)
{
	return updateDeflationOfContinuationValues_gpu(   thrust::raw_pointer_cast(&deflatedNextStepValues_device[0]),  
		thrust::raw_pointer_cast(&numeraireValues_device[0]),
		paths,
		newStepNumber,
		oldStepNumber);
}
/*

double generateRegressionCoefficientsViaLSquadratic_gpu(int numberExerciseDates,
														//														std::vector<float>& products_cube_vec,
														//														std::vector<float>& targets_mat_vec,
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
														)
{
	if (useCrossTerms)
		return generateRegressionCoefficientsViaLS_gpu<quadraticPolynomialCrossDevice>( numberExerciseDates,		
		//								products_cube_vec,
		//								targets_mat_vec,
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
		useCrossTerms,
		testPassed,
		products_gold_cube, // ignored if not in testmode
		targets_gold_mat,// ignored if not in testmode
		coefficients_gold,                                 // ignored if not in testmode
		testTolerance
		);

	else

		return generateRegressionCoefficientsViaLS_gpu<quadraticPolynomialDevice>( numberExerciseDates,		
		//								products_cube_vec,
		//								targets_mat_vec,
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
		testTolerance
		);
}

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
														bool useCrossTerms)
{
	if (useCrossTerms)
		return generateRegressionCoefficientsViaLS_gpu<quadraticPolynomialCrossDevice>( numberExerciseDates,		
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
		normalise
		);

	else
		return generateRegressionCoefficientsViaLS_gpu<quadraticPolynomialDevice>( numberExerciseDates,		
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
		normalise
		);

}

*/
