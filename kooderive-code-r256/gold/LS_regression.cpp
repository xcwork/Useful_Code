
//
//
//                  LS_regression.cpp
//
//
// (c) Mark Joshi 2011,2012,2013
// This code is released under the GNU public licence version 3
/*
The routines are nested as follows. 
1. generateRegressionCoefficientsViaLSQuadratic_gold short arg list
2. generateRegressionCoefficientsViaLSQuadratic_gold long arg list, additional arguments are workspace variables
3. generateRegressionCoefficientsViaLS_gold<quadraticPolynomialGenerator>  generic function to allows different polynomials

4. for each step  : GenerateLSRegressionMatrix_gold      // computes E(X_i X_j) for all i j 
updateDeflatedNextStepValues_gold       // deflates between steps

*/

#include <gold/LS_regression.h>
#include <vector>
#include <gold/MatrixFacade.h>
#include <gold/math/svd_gold.h>
#include <gold/math/typedefs_math_gold.h>
#include <gold/Regression_Selector_concrete_gold.h>

/*
void GenerateLSRegressionMatrixQuadratic_gold( 
	const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
	const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
	const std::vector<Realv>& deflatedNextStepValues_vec, // deflated to current exercise time
	int stepNumber,
	int totalPaths,
	int numberVariablesThisStep,
	MatrixFacade<double>& regressionMatrix,
	std::vector<double>& sumProductsWithContinuationValues,
	std::vector<Realv>& workSpace1_vec,
	std::vector<Realv>& workSpace2_vec,
	std::vector<Realv>& workSpace3_vec,
	std::vector<Realv>& workSpace4_vec,
	bool normalise
	)
{
	quadraticPolynomialGenerator polyGen(numberVariablesThisStep);

	GenerateLSRegressionMatrix_gold<quadraticPolynomialGenerator>( 
		basisFunctionVariables_cube,
		deflatedCashFlowValues_matrix, // deflated to current exercise time
		deflatedNextStepValues_vec, // deflated to current exercise time
		stepNumber,
		totalPaths,
		numberVariablesThisStep,
		polyGen,
		regressionMatrix,
		sumProductsWithContinuationValues,
		workSpace1_vec,
		workSpace2_vec,
		workSpace3_vec,
		workSpace4_vec,
		normalise);


}


void updateDeflatedNextStepValuesQuadratic_gold(const MatrixFacade<double>& regressionWeights,
												Realv AndersenShift,
												const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
												const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
												std::vector<Realv>& deflatedNextStepValues_vec, // deflated to current exercise time
												const MatrixConstFacade<Realv>& deflatedExerciseValue_matrix,
												int stepNumber,
												int totalPaths,
												int numberVariablesThisStep,
												std::vector<Realv>& workSpace1_vec,
												std::vector<Realv>& workSpace2_vec)
{
	quadraticPolynomialGenerator generator(numberVariablesThisStep);

	updateDeflatedNextStepValues_gold(regressionWeights,
		AndersenShift,
		basisFunctionVariables_cube,
		deflatedCashFlowValues_matrix, // deflated to current exercise time
		deflatedNextStepValues_vec, // deflated to current exercise time
		deflatedExerciseValue_matrix,
		stepNumber,
		totalPaths,
		numberVariablesThisStep,
		generator,
		workSpace1_vec,
		workSpace2_vec
		);

}

*/

void updateDeflationOfContinuationValues_gold( std::vector<Realv>& deflatedNextStepValues_vec, 
											  const MatrixFacade<Realv>& numeraireValues_matrix,
											  int paths,
											  int newStepNumber,
											  int oldStepNumber)
{

	if (newStepNumber != oldStepNumber)
		for (int p=0; p < paths; ++p)
		{
			Realv df = numeraireValues_matrix(newStepNumber,p)/numeraireValues_matrix(oldStepNumber,p);
#ifdef _DEBUG
if (fp_isnan(df))
	GenerateError("df is nan");
#endif


			deflatedNextStepValues_vec[p] *= df;
		}
}

/*

template<class T>
double generateRegressionCoefficientsViaLS_gold(int numberExerciseDates,
												std::vector<double>& products_cube_vec,
												std::vector<double>& targets_mat_vec,
												std::vector<double>& coefficients_vec,
												const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
												const std::vector<int>& basisVariablesEachStep,
												int maxBasisVariables,
												const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
												const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, // deflated to current exercise time
												const MatrixFacade<Realv>& numeraireValues_matrix,
												const std::vector<int>& deflation_locations_vec,
												int paths,
												bool normalise,
												std::vector<Realv>& nextStepWorkspace_vec,
												std::vector<Realv>& basisVariableWorkspace_vec,
												std::vector<Realv>& basisFunctionsWorkspace_vec,
												std::vector<double>& regressionMatrixWorkspace_vec,
												std::vector<double>& sumProductsWorkspace_vec,
												std::vector<double>& meansWorkspace_vec,
												std::vector<double>& sdsWorkspace_vec)
{
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

	coefficients_vec.resize(maxDataPoints*numberExerciseDates);
	std::vector<Realv> coefficients_step_vec(maxDataPoints);

	MatrixFacade<double> regressCoeffs_mat(&coefficients_vec[0],numberExerciseDates,maxDataPoints);

	products_cube_vec.resize(maxDataPoints*maxDataPoints*numberExerciseDates,0.0);
	targets_mat_vec.resize(maxDataPoints*numberExerciseDates,0.0);
	CubeFacade<double> products_cube(products_cube_vec,numberExerciseDates,maxDataPoints,maxDataPoints);
	MatrixFacade<double> targets_mat(targets_mat_vec,numberExerciseDates,maxDataPoints);

	T generator(oldnumberBasisVariables);

	int numberDataPoints=generator.numberDataPoints();
	for (int step= numberExerciseDates-1;  step >=0;--step)
	{
		int numberBasisVariables = basisVariablesEachStep[step];



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

		GenerateLSRegressionMatrix_gold( basisFunctionVariables_cube,
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
			normalise
			);



		for (int i=0; i < numberDataPoints; ++i)
		{
			targets_mat(step,i) = sumProductsWorkspace_vec[i];
			for (int j=0; j < numberDataPoints; ++j)
				products_cube(step,i,j) = regressionMatrix_cmat(i,j);
		}



		Real_d tolerance = 1E-8;

		svdSolver.SVDSolve(regressionMatrix_cmat, 
			sumProductsWorkspace_vec, 
			coefficients_step_vec, tolerance);


			if (normalise)
		{
			for (int i=1; i < numberDataPoints; ++i)
			{
				coefficients_step_vec[0] -= coefficients_step_vec[i]*meansWorkspace_vec[i]/sdsWorkspace_vec[i];
			}

			for (int i=1; i < numberDataPoints; ++i)
			{
				coefficients_step_vec[i] /= sdsWorkspace_vec[i];
			}

		}



		for (size_t i=0; i < coefficients_step_vec.size(); ++i)
			regressCoeffs_mat(step,i) = coefficients_step_vec[i];


		Realv andersenShift =0.0f;

		//debugDumpVector(nextStepWorkspace_vec,"nextStepWorkspace_vec pre");

		updateDeflatedNextStepValues_gold(regressCoeffs_mat,
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
			deflation_locations_vec[step-1],
			deflation_locations_vec[step]);

	}



	double res=0.0;

	for (int i=0; i < paths; ++i)
		res+= nextStepWorkspace_vec[i]*numeraireValues_matrix(0,i);

	res/=paths;

	return res;
}



double generateRegressionCoefficientsViaLSQuadratic_gold(int numberExerciseDates,
														 std::vector<double>& products_cube_vec,
														 std::vector<double>& targets_mat_vec,
														 std::vector<double>& coefficients,
														 const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
														 const std::vector<int>& basisVariablesEachStep,
														 int maxBasisVariables,
														 const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
														 const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, // deflated to current exercise time
														 const MatrixFacade<Realv>& numeraireValues_matrix,
														 const std::vector<int>& deflation_locations_vec,
														 int paths,
														 std::vector<Realv>& nextStepWorkspace_vec,
														 std::vector<Realv>& basisVariableWorkspace_vec,
														 std::vector<Realv>& basisFunctionsWorkspace_vec,
														 std::vector<double>& regressionMatrixWorkspace_vec,
														 std::vector<double>& sumProductsWorkspace_vec,
														 std::vector<double>& means_vec,
														 std::vector<double>& sds_vec,
														 bool normalise,
														 bool useCrossTerms)
{

	if (useCrossTerms)

		return   generateRegressionCoefficientsViaLS_gold<quadraticPolynomialCrossGenerator>(numberExerciseDates,
		products_cube_vec,
		targets_mat_vec,
		coefficients,
		basisFunctionVariables_cube,
		basisVariablesEachStep,
		maxBasisVariables,
		deflatedCashFlowValues_matrix, // deflated to current exercise time
		deflatedExerciseValues_matrix, // deflated to current exercise time
		numeraireValues_matrix,
		deflation_locations_vec,
		paths,
		normalise,
		nextStepWorkspace_vec,
		basisVariableWorkspace_vec,
		basisFunctionsWorkspace_vec,
		regressionMatrixWorkspace_vec,
		sumProductsWorkspace_vec,
		means_vec,
		sds_vec);
	else
		return   generateRegressionCoefficientsViaLS_gold<quadraticPolynomialGenerator>(numberExerciseDates,
		products_cube_vec,
		targets_mat_vec,
		coefficients,
		basisFunctionVariables_cube,
		basisVariablesEachStep,
		maxBasisVariables,
		deflatedCashFlowValues_matrix, // deflated to current exercise time
		deflatedExerciseValues_matrix, // deflated to current exercise time
		numeraireValues_matrix,
		deflation_locations_vec,
		paths,
		normalise,
		nextStepWorkspace_vec,
		basisVariableWorkspace_vec,
		basisFunctionsWorkspace_vec,
		regressionMatrixWorkspace_vec,
		sumProductsWorkspace_vec,
		means_vec,
		sds_vec);

}


double generateRegressionCoefficientsViaLSQuadratic_gold(int numberExerciseDates,
														 std::vector<double>& products_cube_vec,
														 std::vector<double>& targets_mat_vec,
														 std::vector<double>& coefficients,
														 const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
														 const std::vector<int>& basisVariablesEachStep,
														 int maxBasisVariables,
														 const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
														 const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, // deflated to current exercise time
														 const MatrixFacade<Realv>& numeraireValues_matrix,
														 const std::vector<int>& deflation_locations_vec,
														 int paths,
														 bool normalise,
														 bool useCrossTerms)
{


	if (useCrossTerms)
	{

		quadraticPolynomialCrossGenerator generator(maxBasisVariables);

		std::vector<Realv> nextStepWorkspace_vec(paths);
		std::vector<Realv> basisVariableWorkspace_vec(maxBasisVariables);
		std::vector<Realv> basisFunctionsWorkspace_vec(generator.numberDataPoints());
		std::vector<double> regressionMatrixWorkspace_vec(generator.numberDataPoints()*generator.numberDataPoints());
		std::vector<double> sumProductsWorkspace_vec(generator.numberDataPoints());
		std::vector<double> means_vec(generator.numberDataPoints());
		std::vector<double> sds_vec(generator.numberDataPoints());


		return generateRegressionCoefficientsViaLSQuadratic_gold( numberExerciseDates,
			products_cube_vec,
			targets_mat_vec,
			coefficients,
			basisFunctionVariables_cube,
			basisVariablesEachStep,
			maxBasisVariables,
			deflatedCashFlowValues_matrix, // deflated to current exercise time
			deflatedExerciseValues_matrix, // deflated to current exercise time
			numeraireValues_matrix,
			deflation_locations_vec,
			paths,
			nextStepWorkspace_vec,
			basisVariableWorkspace_vec,
			basisFunctionsWorkspace_vec,
			regressionMatrixWorkspace_vec,
			sumProductsWorkspace_vec,
			means_vec,
			sds_vec,
			normalise,
			useCrossTerms);
	}
	else
	{
		quadraticPolynomialGenerator generator(maxBasisVariables);

		std::vector<Realv> nextStepWorkspace_vec(paths);
		std::vector<Realv> basisVariableWorkspace_vec(maxBasisVariables);
		std::vector<Realv> basisFunctionsWorkspace_vec(generator.numberDataPoints());
		std::vector<double> regressionMatrixWorkspace_vec(generator.numberDataPoints()*generator.numberDataPoints());
		std::vector<double> sumProductsWorkspace_vec(generator.numberDataPoints());
		std::vector<double> means_vec(generator.numberDataPoints());
		std::vector<double> sds_vec(generator.numberDataPoints());


		return generateRegressionCoefficientsViaLSQuadratic_gold( numberExerciseDates,
			products_cube_vec,
			targets_mat_vec,
			coefficients,
			basisFunctionVariables_cube,
			basisVariablesEachStep,
			maxBasisVariables,
			deflatedCashFlowValues_matrix, // deflated to current exercise time
			deflatedExerciseValues_matrix, // deflated to current exercise time
			numeraireValues_matrix,
			deflation_locations_vec,
			paths,
			nextStepWorkspace_vec,
			basisVariableWorkspace_vec,
			basisFunctionsWorkspace_vec,
			regressionMatrixWorkspace_vec,
			sumProductsWorkspace_vec,
			means_vec,
			sds_vec,
			normalise,
			useCrossTerms);
	}

}


double SecondPassPriceLSAUsingAggregatedFlowsQuadratic_gold(int numberExerciseDates,
															const MatrixConstFacade<double>& LS_coefficients_mat,
															const std::vector<double>& andersenShifts_vec,
															const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
															const std::vector<int>& basisVariablesEachStep,
															int maxBasisVariables,
															const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
															const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, // deflated to current exercise time
															const MatrixFacade<Realv>& numeraireValues_matrix,
															const std::vector<int>& deflationLocations_vec,
															int paths,
															bool useCrossTerms)
{
	if (useCrossTerms)
		return SecondPassPriceLSAUsingAggregatedFlows_gold<quadraticPolynomialCrossGenerator>( numberExerciseDates,
		LS_coefficients_mat,
		andersenShifts_vec,
		basisFunctionVariables_cube,
		basisVariablesEachStep,
		maxBasisVariables,
		deflatedCashFlowValues_matrix, // deflated to current exercise time
		deflatedExerciseValues_matrix, // deflated to current exercise time
		numeraireValues_matrix,
		paths,
		deflationLocations_vec);
	else

		return SecondPassPriceLSAUsingAggregatedFlows_gold<quadraticPolynomialGenerator>( numberExerciseDates,
		LS_coefficients_mat,
		andersenShifts_vec,
		basisFunctionVariables_cube,
		basisVariablesEachStep,
		maxBasisVariables,
		deflatedCashFlowValues_matrix, // deflated to current exercise time
		deflatedExerciseValues_matrix, // deflated to current exercise time
		numeraireValues_matrix,
		paths,
		deflationLocations_vec);
}


void oneStepUpdateLS_gold(int paths,
						  int exerciseNumber,
						  const MatrixConstFacade<Realv>& basisFunctions_mat, // just the function values for this step
						  int numberBasisFunctions,
						  const MatrixConstFacade<Realv>&  basis_weights_mat, // the weights for all steps
						  std::vector<Realv>& continuationValues_vec,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
						  const MatrixConstFacade<Realv>&  cashFlowValues_mat,      // the cash flow values for all steps deflated to current exercise time
						  const MatrixConstFacade<Realv>&  exerciseValues_mat,      // the exerciseValues for all steps deflated to current exercise time
						  const std::vector<Realv>& AndersenShift)
{
	for (int p=0; p < paths; ++p)
	{
		double estContValue = AndersenShift[exerciseNumber];

		for (int i=0; i < numberBasisFunctions; ++i)
			estContValue += basisFunctions_mat(i,p)*basis_weights_mat(exerciseNumber,i);

		Realv exValue = exerciseValues_mat(exerciseNumber,p);

		if ( estContValue < exValue)
			continuationValues_vec[p] = exValue;
		else
			continuationValues_vec[p] +=  cashFlowValues_mat(exerciseNumber,p);


	}

}

*/
