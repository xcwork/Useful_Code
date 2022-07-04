

//
//
//                  MultiLS_regression.cpp
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3
/*
*/

#include <gold/MultiLS_regression.h>
#include <vector>
#include <gold/MatrixFacade.h>
#include <gold/math/svd_gold.h>
#include <gold/math/typedefs_math_gold.h>
#include <gold/Regression_Selector_concrete_gold.h>



template<class T>
double generateRegressionCoefficientsViaLSMulti_gold(int numberExerciseDates,
													 std::vector<double>& products_cube_vec,
													 std::vector<double>& targets_mat_vec,
													 CubeFacade<double>& regression_coefficients_cube,
													 MatrixFacade<double>& lowerCuts_mat,
													 MatrixFacade<double>& upperCuts_mat,
													 CubeFacade<Realv>& means_variables_cube,
													 CubeFacade<Realv>& sds_variables_cube,
													 bool normalise,
													 const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
													 const std::vector<int>& basisVariablesEachStep,
													 int maxBasisVariables,
													 const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
													 const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, // deflated to current exercise time
													 const MatrixFacade<Realv>& numeraireValues_matrix,
												//	 const std::vector<int>& deflation_locations_vec,
													 int paths,
													 int maxNumberOfExtraRegressions,
													 int lowerPathCutoff,
													 RegressionSelectorGold& regressionSelector,
													 std::vector<Realv>& nextStepWorkspace_vec,
													 std::vector<Realv>& basisVariableWorkspace_vec,
													 std::vector<Realv>& basisFunctionsWorkspace_vec,
													 std::vector<double>& regressionMatrixWorkspace_vec,
													 std::vector<double>& sumProductsWorkspace_vec,
													 std::vector<double>& meansWorkspace_vec,
													 std::vector<double>& sdsWorkspace_vec)
{

	//debugDumpCube<double>(basisFunctionVariables_cube,"basisFunctionVariables_cube");
	//debugDumpMatrix(deflatedCashFlowValues_matrix,"deflatedCashFlowValues_matrix");

 //   debugDumpMatrix(numeraireValues_matrix,"numeraireValues_matrix");

	nextStepWorkspace_vec.resize(0);
	nextStepWorkspace_vec.resize(paths,0.0f);
	basisVariableWorkspace_vec.resize(maxBasisVariables);

	int maxDataPoints=-1;
	{
		T generatorM(maxBasisVariables);
		maxDataPoints = generatorM.numberDataPoints();
	}

	basisFunctionsWorkspace_vec.resize(maxDataPoints);

	T generatorLast(basisVariablesEachStep[numberExerciseDates-1]);

	SVDSquareMatrix svdSolver(generatorLast.numberDataPoints());
	int oldnumberBasisVariables = basisVariablesEachStep[numberExerciseDates-1];

	regressionMatrixWorkspace_vec.resize(maxDataPoints*maxDataPoints);

	sumProductsWorkspace_vec.resize(maxDataPoints);
	int maxNumberOfRegressions = maxNumberOfExtraRegressions+1;
	std::vector<Realv> coefficients_step_vec(maxDataPoints);


	products_cube_vec.resize(maxDataPoints*maxDataPoints*numberExerciseDates,0.0);
	targets_mat_vec.resize(maxDataPoints*numberExerciseDates,0.0);
	CubeFacade<double> products_cube(products_cube_vec,numberExerciseDates,maxDataPoints,maxDataPoints);
	MatrixFacade<double> targets_mat(targets_mat_vec,numberExerciseDates,maxDataPoints);

	T generator(oldnumberBasisVariables);

	int numberDataPoints=generator.numberDataPoints();

	std::vector<int> preValid(paths,1);

	std::vector<int> valid(paths,0);
	std::vector<double> netContinuations(paths);
	std::vector<double> workSpace1_vec;
	std::vector<double> workSpace2_vec;

	std::vector<Realv> means_vec(maxBasisVariables,0.0);
	std::vector<Realv> sds_vec(maxBasisVariables,1.0);


	for (int step= numberExerciseDates-1;  step >=0;--step)
	{
		int numberBasisVariables = basisVariablesEachStep[step];
		std::fill(preValid.begin(),preValid.end(),1);



		if (numberBasisVariables != oldnumberBasisVariables)
		{
			T generatorStep(numberBasisVariables);
			SVDSquareMatrix tmp(generatorStep.numberDataPoints());
			svdSolver=tmp;
			numberDataPoints = generatorStep.numberDataPoints();
			generator = generatorStep;

		}

		MatrixFacade<double> regressionMatrix_mat(&regressionMatrixWorkspace_vec[0],numberDataPoints,numberDataPoints);
		MatrixConstFacade<double> regressionMatrix_cmat(&regressionMatrixWorkspace_vec[0],numberDataPoints,numberDataPoints);
		sumProductsWorkspace_vec.resize(numberDataPoints);
		coefficients_step_vec.resize(numberDataPoints);


		if (normalise)
			GenerateLSRegressionMatrixSelectivelyNormalisingVariables_gold( basisFunctionVariables_cube,
			deflatedCashFlowValues_matrix, // deflated to current exercise time
			nextStepWorkspace_vec, // deflated to current exercise time
			step,
			paths,
			numberBasisVariables,
			generator,
			regressionMatrix_mat,
			sumProductsWorkspace_vec,
			basisVariableWorkspace_vec,
			basisFunctionsWorkspace_vec,
			meansWorkspace_vec,
			sdsWorkspace_vec,
			means_vec,
			sds_vec,
			preValid,
			true
			);


		else

			GenerateLSRegressionMatrixSelectively_gold( basisFunctionVariables_cube,
			deflatedCashFlowValues_matrix, // deflated to current exercise time
			nextStepWorkspace_vec, // deflated to current exercise time
			step,
			paths,
			numberBasisVariables,
			generator,
			regressionMatrix_mat,
			sumProductsWorkspace_vec,
			basisVariableWorkspace_vec,
			basisFunctionsWorkspace_vec,
			meansWorkspace_vec,
			sdsWorkspace_vec,
			preValid,
			true
			);

		for (int i=0; i < numberBasisVariables; ++i)
		{
			means_variables_cube(step,0,i) =means_vec[i];
			sds_variables_cube(step,0,i) =sds_vec[i];
		}

//		debugDumpMatrix(regressionMatrix_cmat,"regressionMatrix_cmat");
////		debugDumpVector(sumProductsWorkspace_vec,"sumProductsWorkspace_vec");

		Real_d tolerance = 1E-8;

		svdSolver.SVDSolve(regressionMatrix_cmat, 
			sumProductsWorkspace_vec, 
			coefficients_step_vec, tolerance);

	//	std::cout << "condition number " << svdSolver.GetConditionOfLastSolve() << "\n";

		for (size_t i=0; i < coefficients_step_vec.size(); ++i)
			regression_coefficients_cube(step,0,i) = coefficients_step_vec[i];

	//	debugDumpVector(coefficients_step_vec,"coefficients_step_vec");


		double lowerCutOff, upperCutOff;

		// first regression has been done 

		computedNetContinuationValues_gold(regression_coefficients_cube,
			0, //depth
			0.0, // shift
			basisFunctionVariables_cube,
			//  deflatedCashFlowValues_matrix, // deflated to current exercise time
			//deflatedNextStepValues_vec, // deflated to current exercise time
			deflatedExerciseValues_matrix,
			step,
			paths,
			numberBasisVariables,
			preValid,
			means_vec,
			sds_vec,
			generator,
			workSpace1_vec,
			workSpace2_vec,
			netContinuations.begin()
			);

//		debugDumpVector(netContinuations,"netContinuations");

        double maxNet = *std::max_element(netContinuations.begin(),netContinuations.end());
        double minNet = *std::min_element(netContinuations.begin(),netContinuations.end());

        int pathsAfter;

        if (maxNet <=0.0 || minNet >=0.0)
        {
            // no paths have unclear decision
            pathsAfter=0;


        }
        else
		 pathsAfter = regressionSelector.Select(0,
			netContinuations.begin(), 
			netContinuations.end(),
			valid.begin(),
			preValid.begin(),
			paths,
			lowerCutOff,
			upperCutOff);


		//		debugDumpVector(valid,"valid");


		if (pathsAfter <=lowerPathCutoff)
		{
			lowerCuts_mat(step,0) = 0.0;
			upperCuts_mat(step,0) = 0.0;

		}
		else
		{
			lowerCuts_mat(step,0) = lowerCutOff;
			upperCuts_mat(step,0) = upperCutOff;


			int depth =0;
			int pathsBefore;

			if (maxNumberOfRegressions > 1)
				do
				{
					++depth;
					preValid = valid;
					pathsBefore	= pathsAfter;

			//		std::cout << "paths pathsAfter, " << pathsAfter << "\n";


					if (normalise)
						GenerateLSRegressionMatrixSelectivelyNormalisingVariables_gold( basisFunctionVariables_cube,
						deflatedCashFlowValues_matrix, // deflated to current exercise time
						nextStepWorkspace_vec, // deflated to current exercise time
						step,
						paths,
						numberBasisVariables,
						generator,
						regressionMatrix_mat,
						sumProductsWorkspace_vec,
						basisVariableWorkspace_vec,
						basisFunctionsWorkspace_vec,
						meansWorkspace_vec,
						sdsWorkspace_vec,
						means_vec,
						sds_vec,
						preValid,
						false
						);


					else
						GenerateLSRegressionMatrixSelectively_gold( basisFunctionVariables_cube,
						deflatedCashFlowValues_matrix, // deflated to current exercise time
						nextStepWorkspace_vec, // deflated to current exercise time
						step,
						paths,
						numberBasisVariables,
						generator,
						regressionMatrix_mat,
						sumProductsWorkspace_vec,
						basisVariableWorkspace_vec,
						basisFunctionsWorkspace_vec,
						meansWorkspace_vec,
						sdsWorkspace_vec,
						preValid,
						false
						);

				//			debugDumpMatrix(regressionMatrix_cmat,"regressionMatrix_cmat in multiroutine");
					//		debugDumpVector(sumProductsWorkspace_vec,"sumProductsWorkspace_vec multiroutine");
		//					debugDumpVector(nextStepWorkspace_vec,"nextStepWorkspace_vec");
					//			   SVDSquareMatrix svdSolver2(sumProductsWorkspace_vec.size());

					svdSolver.SVDSolve(regressionMatrix_cmat, 
						sumProductsWorkspace_vec, 
						coefficients_step_vec, tolerance);
					//				debugDumpVector(coefficients_step_vec,"coefficients_step_vec multiroutine");

					
	      //       	std::cout << "condition number " << svdSolver.GetConditionOfLastSolve() << "\n";

					for (size_t i=0; i < coefficients_step_vec.size(); ++i)
					{
						regression_coefficients_cube(step,depth,i) = coefficients_step_vec[i];

					}
					for (int i=0; i < numberBasisVariables; ++i)
					{
						means_variables_cube(step,depth,i) = means_vec[i];
						sds_variables_cube(step,depth,i) = sds_vec[i];
					}


					//		debugDumpVector(coefficients_step_vec,"coefficients_step_vec");

					double lowerCutOff, upperCutOff;

					// nested regression has been done 

					computedNetContinuationValues_gold(regression_coefficients_cube,
						depth, //depth
						0.0, // shift
						basisFunctionVariables_cube,
						//  deflatedCashFlowValues_matrix, // deflated to current exercise time
						//deflatedNextStepValues_vec, // deflated to current exercise time
						deflatedExerciseValues_matrix,
						step,
						paths,
						numberBasisVariables,
						preValid,
						means_vec,
						sds_vec,
						generator,
						workSpace1_vec,
						workSpace2_vec,
						netContinuations.begin()
						);

					//	debugDumpVector(netContinuations,"netContinuations");


					pathsAfter = regressionSelector.Select(depth,
						netContinuations.begin(), 
						netContinuations.end(),
						valid.begin(),
						preValid.begin(),
						pathsBefore,
						lowerCutOff,
						upperCutOff);

			
					//	debugDumpVector(valid,"valid");

					if (pathsAfter <= lowerPathCutoff)
					{
						 lowerCuts_mat(step,depth) = 0.0;
					     upperCuts_mat(step,depth) = 0.0;

					}
					else
					{
						lowerCuts_mat(step,depth) = lowerCutOff;
						upperCuts_mat(step,depth) = upperCutOff;
					}


				}
				while ((depth+1< maxNumberOfRegressions) && (pathsAfter > lowerPathCutoff));

		}


		Realv andersenShift =0.0f;
		updateDeflatedNextStepValuesMulti_gold(regression_coefficients_cube,
			lowerCuts_mat,
			upperCuts_mat,
			means_variables_cube,
			sds_variables_cube,
			maxNumberOfRegressions,
			andersenShift,
			basisFunctionVariables_cube,
			deflatedCashFlowValues_matrix, // deflated to current exercise time
			nextStepWorkspace_vec, // deflated to current exercise time
			deflatedExerciseValues_matrix,
			step,
			paths,
			numberBasisVariables,
			generator,
			basisVariableWorkspace_vec,
			basisFunctionsWorkspace_vec
			);

		//	debugDumpVector(nextStepWorkspace_vec,"nextStepWorkspace_vec post");


		if (step >0)
			updateDeflationOfContinuationValues_gold(nextStepWorkspace_vec, 
			numeraireValues_matrix,
			paths,
			step-1,
			step);

	}



	double res=0.0;

	for (int i=0; i < paths; ++i)
		res+= nextStepWorkspace_vec[i]/numeraireValues_matrix(0,i);

//	debugDumpVector(nextStepWorkspace_vec,"nextStepWorkspace_vec");

	res/=paths;

	return res;
}





double generateRegressionCoefficientsViaLSMultiQuadratic_gold(int numberExerciseDates,
															  std::vector<double>& products_cube_vec,
															  std::vector<double>& targets_mat_vec,
															  CubeFacade<double>& regression_coefficients_cube,
															  MatrixFacade<double>& lowerCuts_mat,
															  MatrixFacade<double>& upperCuts_mat,
															  CubeFacade<Realv>& means_variables_cube,
															  CubeFacade<Realv>& sds_variables_cube,
															  bool normalise,
															  const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
															  const std::vector<int>& basisVariablesEachStep,
															  int maxBasisVariables,
															  const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
															  const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, // deflated to current exercise time
															  const MatrixFacade<Realv>& numeraireValues_matrix,
												//			  const std::vector<int>& deflation_locations_vec,
															  int paths,
															  int maxNumberOfExtraRegressions,
															  int lowerPathCutoff,
															  RegressionSelectorGold& regressionSelector,
															  std::vector<Realv>& nextStepWorkspace_vec,
															  std::vector<Realv>& basisVariableWorkspace_vec,
															  std::vector<Realv>& basisFunctionsWorkspace_vec,
															  std::vector<double>& regressionMatrixWorkspace_vec,
															  std::vector<double>& sumProductsWorkspace_vec,
															  std::vector<double>& meansWorkspace_vec,
															  std::vector<double>& sdsWorkspace_vec, 
															  bool useCrossTerms)
{
	if (useCrossTerms)
	{
		return generateRegressionCoefficientsViaLSMulti_gold<quadraticPolynomialCrossGenerator>(
			numberExerciseDates,
			products_cube_vec,
			targets_mat_vec,
			regression_coefficients_cube,
			lowerCuts_mat,
			upperCuts_mat,
			means_variables_cube,
			sds_variables_cube,
			normalise,
			basisFunctionVariables_cube,
			basisVariablesEachStep,
			maxBasisVariables,
			deflatedCashFlowValues_matrix, // deflated to current exercise time
			deflatedExerciseValues_matrix, // deflated to current exercise time
			numeraireValues_matrix,
		//	deflation_locations_vec,
			paths,
			maxNumberOfExtraRegressions,
			lowerPathCutoff,
			regressionSelector,
			nextStepWorkspace_vec,
			basisVariableWorkspace_vec,
			basisFunctionsWorkspace_vec,
			regressionMatrixWorkspace_vec,
			sumProductsWorkspace_vec,
			meansWorkspace_vec,
			sdsWorkspace_vec
			);
	}
	else
	{
		return generateRegressionCoefficientsViaLSMulti_gold<quadraticPolynomialGenerator>(
			numberExerciseDates,
			products_cube_vec,
			targets_mat_vec,
			regression_coefficients_cube,
			lowerCuts_mat,
			upperCuts_mat,
			means_variables_cube,
			sds_variables_cube,
			normalise,
			basisFunctionVariables_cube,
			basisVariablesEachStep,
			maxBasisVariables,
			deflatedCashFlowValues_matrix, // deflated to current exercise time
			deflatedExerciseValues_matrix, // deflated to current exercise time
			numeraireValues_matrix,
	//		deflation_locations_vec,
			paths,
			maxNumberOfExtraRegressions,
			lowerPathCutoff,
			regressionSelector,
			nextStepWorkspace_vec,
			basisVariableWorkspace_vec,
			basisFunctionsWorkspace_vec,
			regressionMatrixWorkspace_vec,
			sumProductsWorkspace_vec,
			meansWorkspace_vec,
			sdsWorkspace_vec
			);
	}


}




double generateRegressionCoefficientsViaLSMultiQuadratic_gold(int numberExerciseDates,
															  std::vector<double>& products_cube_vec,
															  std::vector<double>& targets_mat_vec,
															  CubeFacade<double>& regression_coefficients_cube,
															  MatrixFacade<double>& lowerCuts_mat,
															  MatrixFacade<double>& upperCuts_mat,
															  CubeFacade<Realv>& means_variables_cube,
															  CubeFacade<Realv>& sds_variables_cube,
															  bool normalise,
															  const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
															  const std::vector<int>& basisVariablesEachStep,
															  int maxBasisVariables,
															  const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
															  const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, // deflated to current exercise time
															  const MatrixFacade<Realv>& numeraireValues_matrix,
														//	  const std::vector<int>& deflation_locations_vec,
															  int paths,
															  int maxNumberOfExtraRegressions,
															  int lowerPathCutoff,
															  RegressionSelectorGold& regressionSelector,
															  bool useCrossTerms)
{
	std::vector<Realv> nextStepWorkspace_vec;
	std::vector<Realv> basisVariableWorkspace_vec;
	std::vector<Realv>  basisFunctionsWorkspace_vec;
	std::vector<double>  regressionMatrixWorkspace_vec;
	std::vector<double>  sumProductsWorkspace_vec;
	std::vector<double> meansWorkspace_vec;
	std::vector<double>  sdsWorkspace_vec;


	if (useCrossTerms)
	{
		return generateRegressionCoefficientsViaLSMulti_gold<quadraticPolynomialCrossGenerator>(
			numberExerciseDates,
			products_cube_vec,
			targets_mat_vec,
			regression_coefficients_cube,
			lowerCuts_mat,
			upperCuts_mat,
			means_variables_cube,
			sds_variables_cube,
			normalise,
			basisFunctionVariables_cube,
			basisVariablesEachStep,
			maxBasisVariables,
			deflatedCashFlowValues_matrix, // deflated to current exercise time
			deflatedExerciseValues_matrix, // deflated to current exercise time
			numeraireValues_matrix,
		//	deflation_locations_vec,
			paths,
			maxNumberOfExtraRegressions,
			lowerPathCutoff,
			regressionSelector,
			nextStepWorkspace_vec,
			basisVariableWorkspace_vec,
			basisFunctionsWorkspace_vec,
			regressionMatrixWorkspace_vec,
			sumProductsWorkspace_vec,
			meansWorkspace_vec,
			sdsWorkspace_vec
			);
	}
	else
	{
		return generateRegressionCoefficientsViaLSMulti_gold<quadraticPolynomialGenerator>(
			numberExerciseDates,
			products_cube_vec,
			targets_mat_vec,
			regression_coefficients_cube,
			lowerCuts_mat,
			upperCuts_mat,
			means_variables_cube,
			sds_variables_cube,
			normalise,
			basisFunctionVariables_cube,
			basisVariablesEachStep,
			maxBasisVariables,
			deflatedCashFlowValues_matrix, // deflated to current exercise time
			deflatedExerciseValues_matrix, // deflated to current exercise time
			numeraireValues_matrix,
	//		deflation_locations_vec,
			paths,
			maxNumberOfExtraRegressions,
			lowerPathCutoff,
			regressionSelector,
			nextStepWorkspace_vec,
			basisVariableWorkspace_vec,
			basisFunctionsWorkspace_vec,
			regressionMatrixWorkspace_vec,
			sumProductsWorkspace_vec,
			meansWorkspace_vec,
			sdsWorkspace_vec
			);
	}


}