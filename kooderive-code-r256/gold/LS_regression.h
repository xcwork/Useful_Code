
//
//
//                  LS_regression.h
//
//
// (c) Mark Joshi 2011, 2012, 2013
// This code is released under the GNU public licence version 3
/*

the routines here are for estimating early exercise strategies via regression
no assumption about the model is made. 

Note that the data should have been preprocessed to map to individual exercise
times. So the time/steps dimension of the data is the number of exercise times not the
number of steps in the original model. 

the main inputs are 

basisFunctionVariables_cube - a cube of values of market variables : path, variable number, time step
basisVariablesEachStep -- how many variables there are for each step
deflatedCashFlowValues_matrix -- the deflated cash flows after exercise time that must be received if exercise does not occur, 
deflation to current exercise time
deflatedExerciseValues_matrix -- the deflated exercise value if exercised now -- deflation to current exercise time
numeraireValues_matrix -- the value of the ctsly compounding money market -- paths \times steps 
*/
#ifndef LS_REGRESSION_H
#define LS_REGRESSION_H

#include <vector>
#include <gold/MatrixFacade.h>
#include <float.h>
#include <gold/math/fp_utilities.h>
#include <gold/math/typedefs_math_gold.h>
#include <iostream>
#include <cmath>

// main routine for carrying out full regression back to 0
/*
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
														 bool useCrossTerms);

// this does the exercise decisions for one step of the first pass of an LS regression for early exercise

void oneStepUpdateLS_gold(int paths,
						  int exerciseNumber,
						  const MatrixConstFacade<Realv>& basisFunctions_mat, // just the function values for this step
						  int numberBasisFunctions,
						  const MatrixConstFacade<Realv>&  basis_weights_mat, // the weights for all steps
						  std::vector<Realv>& continuationValues_vec,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
						  const MatrixConstFacade<Realv>&  cashFlowValues_mat,      // the cash flow values for all steps deflated to current exercise time
						  const MatrixConstFacade<Realv>&  exerciseValues_mat,      // the exerciseValues for all steps deflated to current exercise time
						  const std::vector<Realv>& AndersenShift);
*/
// following routines are used by the main one
// provided separately 

// typically oldStepNumber = newStepNumber+1, and so we are discounting across 1 step
void updateDeflationOfContinuationValues_gold( std::vector<Realv>& deflatedNextStepValues_vec, 
											  const MatrixFacade<Realv>& numeraireValues_matrix,
											  int paths,
											  int newStepNumber,
											  int oldStepNumber);

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
	std::vector<Realv>& workSpace2_vec
	);
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
												std::vector<Realv>& workSpace2_vec);



// calls the templatized version with a quadratic parameter
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
															bool useCrossTerms);
															*/
// small class for turning basis variables into polynomials. 
// eg given the values a b c generate a vector
// 1 a b c a^2  b^2  c^2 
// the number of variables is the main input 
// no cross terms
class quadraticPolynomialGenerator
{

private:

	int numberVariables_;
	int total_;

public:

	quadraticPolynomialGenerator(int numberVariables=0) : numberVariables_(numberVariables)

	{
		total_=1+2*numberVariables_;

	}

	int numberDataPoints() const
	{
		return total_;

	}

	template<class T>
	void writeData(const std::vector<T>& values, std::vector<T>& polyValues) const
	{
		polyValues[0]=static_cast<T>(1.0);
		int j=1;
		for (int i=0; i < numberVariables_; ++i,++j)
		{
			T x = values[i];
			polyValues[j] = values[i];
			++j;
			polyValues[j] = x*x;
		}

	}


};
/*
1 x x^2 xy y y^2 yz z z^2 zx , only does cross term with next variable so if four variables some are missing
*/
class quadraticPolynomialCrossGenerator
{

private:

	int numberVariables_;
	int total_;

public:

	quadraticPolynomialCrossGenerator(int numberVariables=0) : numberVariables_(numberVariables)

	{
		if (numberVariables ==0)
			total_ = 1;
		else
			if (numberVariables==1)
				total_ = 3;
			else
				if (numberVariables ==2)
					total_ = 6;
				else
					total_=1+3*numberVariables_;

	}

	int numberDataPoints() const
	{
		return total_;

	}

	template<class T>
	void writeData(const std::vector<T>& values, std::vector<T>& polyValues) const
	{
		polyValues[0]=static_cast<T>(1.0);
		int j=1;
		for (int i=0; i < numberVariables_; ++i,++j)
		{
			T x=values[i];
			polyValues[j] = x;
		}
		if (numberVariables_ ==1)
		{
			T x=values[0];
			polyValues[j] = x*x;
			return;
		}

		if (numberVariables_==2)
		{
			T x=values[0];
			T y=values[1];
			polyValues[j] = x*x;
			++j;
			polyValues[j] = x*y;
			++j;
			polyValues[j] = y*y;
			return;
		}

		for (int i=0; i < numberVariables_; ++i,++j)
		{
			T x =values[i];
			int k = (i+1) % numberVariables_;
			T y = values[k];
			polyValues[j] = x*y;

		}


		for (int i=0; i < numberVariables_; ++i,++j)
		{
			T x =values[i];

			polyValues[j] = x*x;

		}


	}


};



/// creates the matrix and target vector to be solved for the regression
template<class T>
void GenerateLSRegressionMatrix_gold( 
	const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
	const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
	const std::vector<Realv>& deflatedNextStepValues_vec, // deflated to current exercise time
	int stepNumber,
	int totalPaths,
	int numberVariablesThisStep,
	T& generator,
	MatrixFacade<double>& regressionMatrix,
	std::vector<double>& sumProductsWithContinuationValues,
	std::vector<Realv>& workSpace1_vec,
	std::vector<Realv>& workSpace2_vec,
	std::vector<Realv>& workSpace3_vec,
	std::vector<Realv>& workSpace4_vec,
	bool normalise
	)
{

	workSpace1_vec.resize(numberVariablesThisStep);
	int numberDataPts= generator.numberDataPoints();
	workSpace2_vec.resize(numberDataPts);
	workSpace3_vec.resize(numberDataPts);
	workSpace4_vec.resize(numberDataPts);

	for (int i=0; i < numberDataPts; ++i)
	{
		sumProductsWithContinuationValues[i]=0.0;
		workSpace3_vec[i]=0.0;
		workSpace4_vec[i]=0.0;
		for (int j=0; j < numberDataPts; ++j)
			regressionMatrix(i,j)=0.0;
	}
	if (normalise) // we are normalising the functions not the variables
	{
		for (int p=0; p < totalPaths; ++p)
		{
			for (int i=0; i < numberVariablesThisStep; ++i) 
			{
				workSpace1_vec[i] = basisFunctionVariables_cube(stepNumber,i,p);
			}
			generator.writeData(workSpace1_vec, workSpace2_vec);

			// compute means and standard deviations 
			for (int i=1; i < numberDataPts; ++i) // implicit assumption is that first basis function is 1 always
			{
				double x= workSpace2_vec[i];
				workSpace3_vec[i] += x;
				workSpace4_vec[i] += x*x;
			}
		}

		for (int i=1; i < numberDataPts; ++i)
		{
			workSpace3_vec[i] /= totalPaths;
			workSpace4_vec[i] /= totalPaths;
			workSpace4_vec[i] -= workSpace3_vec[i]*workSpace3_vec[i];
			workSpace4_vec[i] = sqrt(workSpace4_vec[i]);
		}

		workSpace3_vec[0] =0.0;
		workSpace4_vec[0] =1.0;

		// means and sds are now computed 

		for (int p=0; p < totalPaths; ++p)
		{
			//   std::cout << "\n " << p << ",";
			for (int i=0; i < numberVariablesThisStep; ++i)
			{
				workSpace1_vec[i] = basisFunctionVariables_cube(stepNumber,i,p);
				//         std::cout << workSpace1_vec[i]  << ",";
			}
			generator.writeData(workSpace1_vec, workSpace2_vec);

			//  for (size_t i=0; i < workSpace2_vec.size(); ++i)
			//     {
			//       
			//         std::cout << workSpace2_vec[i]  << ",";
			//     }

			Realv cashValue = deflatedCashFlowValues_matrix(stepNumber,p) + deflatedNextStepValues_vec[p];

			//     std::cout << cashValue << ",";

#ifdef _DEBUG
			if (fp_isnan(cashValue))
				throw("isnan!");
#endif

			for (int i=0; i < numberDataPts; ++i)
			{
				workSpace2_vec[i] -= workSpace3_vec[i];
				workSpace2_vec[i] /= workSpace4_vec[i];
			}

			for (int i=0; i < numberDataPts; ++i)
			{

				for (int j=0; j <= i; ++j)
					regressionMatrix(i,j) += workSpace2_vec[i]*workSpace2_vec[j];

				sumProductsWithContinuationValues[i] += cashValue*workSpace2_vec[i];
			}

		}

		for (int i=0; i < numberDataPts; ++i)
		{
			regressionMatrix(i,i) /= totalPaths;
			sumProductsWithContinuationValues[i]/=totalPaths;

			for (int j=0; j < i; ++j)
			{
				regressionMatrix(i,j) /=totalPaths;
				regressionMatrix(j,i) = regressionMatrix(i,j);
			}
		}
	}
	else
	{
		for (int p=0; p < totalPaths; ++p)
		{
			//   std::cout << "\n " << p << ",";
			for (int i=0; i < numberVariablesThisStep; ++i)
			{
				workSpace1_vec[i] = basisFunctionVariables_cube(stepNumber,i,p);
				//         std::cout << workSpace1_vec[i]  << ",";
			}
			generator.writeData(workSpace1_vec, workSpace2_vec);

			//  for (size_t i=0; i < workSpace2_vec.size(); ++i)
			//     {
			//       
			//         std::cout << workSpace2_vec[i]  << ",";
			//     }

			Realv cashValue = deflatedCashFlowValues_matrix(stepNumber,p) + deflatedNextStepValues_vec[p];

			//     std::cout << cashValue << ",";

#ifdef _DEBUG
			if (fp_isnan(cashValue))
				throw("isnan!");
#endif

			for (int i=0; i < numberDataPts; ++i)
			{
				for (int j=0; j <= i; ++j)
					regressionMatrix(i,j) += workSpace2_vec[i]*workSpace2_vec[j];

				sumProductsWithContinuationValues[i] += cashValue*workSpace2_vec[i];


			}

		}

		for (int i=0; i < numberDataPts; ++i)
		{
			regressionMatrix(i,i) /= totalPaths;
			sumProductsWithContinuationValues[i]/=totalPaths;

			for (int j=0; j < i; ++j)
			{
				regressionMatrix(i,j) /=totalPaths;
				regressionMatrix(j,i) = regressionMatrix(i,j);
			}

		}

		// std::cout << "\n\n";

	}
}

template<class T>
void GenerateLSRegressionMatrixSelectively_gold( 
	const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
	const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
	const std::vector<Realv>& deflatedNextStepValues_vec, // deflated to current exercise time
	int stepNumber,
	int totalPaths,
	int numberVariablesThisStep,
	T& generator,
	MatrixFacade<double>& regressionMatrix,
	std::vector<double>& sumProductsWithContinuationValues,
	std::vector<Realv>& workSpace1_vec,
	std::vector<Realv>& workSpace2_vec,
	std::vector<Realv>& workSpace3_vec,
	std::vector<Realv>& workSpace4_vec,
	const std::vector<int>& validity,
	bool allValid
	)
{

	workSpace1_vec.resize(numberVariablesThisStep);
	int numberDataPts= generator.numberDataPoints();
	workSpace2_vec.resize(numberDataPts);
	workSpace3_vec.resize(numberDataPts);
	workSpace4_vec.resize(numberDataPts);

	for (int i=0; i < numberDataPts; ++i)
	{
		sumProductsWithContinuationValues[i]=0.0;
		workSpace3_vec[i]=0.0;
		workSpace4_vec[i]=0.0;
		for (int j=0; j < numberDataPts; ++j)
			regressionMatrix(i,j)=0.0;
	}


	int validPaths=0;

	for (int p=0; p < totalPaths; ++p)
	{
		if (allValid || validity[p])
		{
			++validPaths;

			for (int i=0; i < numberVariablesThisStep; ++i)
			{
				workSpace1_vec[i] = basisFunctionVariables_cube(stepNumber,i,p);
			}
			generator.writeData(workSpace1_vec, workSpace2_vec);

			Realv cashValue = deflatedCashFlowValues_matrix(stepNumber,p) + deflatedNextStepValues_vec[p];


#ifdef _DEBUG
			if (fp_isnan(cashValue))
				throw("isnan!");
#endif

			for (int i=0; i < numberDataPts; ++i)
			{
				for (int j=0; j <= i; ++j)
					regressionMatrix(i,j) += workSpace2_vec[i]*workSpace2_vec[j];

				sumProductsWithContinuationValues[i] += cashValue*workSpace2_vec[i];

		//		std::cout << workSpace2_vec[i] << ",";

			}

	//		std::cout << "," << cashValue << "\n";

		}
	}

	for (int i=0; i < numberDataPts; ++i)
	{
		regressionMatrix(i,i) /= validPaths;
		sumProductsWithContinuationValues[i]/=validPaths;

		for (int j=0; j < i; ++j)
		{
			regressionMatrix(i,j) /=validPaths;
			regressionMatrix(j,i) = regressionMatrix(i,j);
		}

	}
//	debugDumpVector(validity,"validity");
//	debugDumpMatrix(MatrixConstFacade<double>(regressionMatrix), "regressionMatrix");
	//debugDumpVector(sumProductsWithContinuationValues, "sumProductsWithContinuationValues");

}


template<class T>
void GenerateLSRegressionMatrixSelectivelyNormalisingVariables_gold( 
	const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
	const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
	const std::vector<Realv>& deflatedNextStepValues_vec, // deflated to current exercise time
	int stepNumber,
	int totalPaths,
	int numberVariablesThisStep,
	T& generator,
	MatrixFacade<double>& regressionMatrix,
	std::vector<double>& sumProductsWithContinuationValues,
	std::vector<Realv>& workSpace1_vec,
	std::vector<Realv>& workSpace2_vec,
	std::vector<Realv>& workSpace3_vec,
	std::vector<Realv>& workSpace4_vec,
	std::vector<Realv>& means_vec,
	std::vector<Realv>& sds_vec,
	const std::vector<int>& validity,
	bool allValid
	)
{

	workSpace1_vec.resize(numberVariablesThisStep);
	int numberDataPts= generator.numberDataPoints();
	workSpace2_vec.resize(numberDataPts);
	workSpace3_vec.resize(numberDataPts);
	workSpace4_vec.resize(numberDataPts);
	
	means_vec.resize(0);
	sds_vec.resize(0);

	means_vec.resize(numberVariablesThisStep,0.0);
	sds_vec.resize(numberVariablesThisStep,0.0);


	// clear data
	for (int i=0; i < numberDataPts; ++i)
	{
		sumProductsWithContinuationValues[i]=0.0;
		workSpace3_vec[i]=0.0;
		workSpace4_vec[i]=0.0;
		for (int j=0; j < numberDataPts; ++j)
			regressionMatrix(i,j)=0.0;
}


	int validPaths=0;

	
	for (int p=0; p < totalPaths; ++p)
	{
		if (allValid || validity[p])
		{
			++validPaths;
			for (int i=0; i < numberVariablesThisStep; ++i)
			{
				 Realv x= basisFunctionVariables_cube(stepNumber,i,p);
				 means_vec[i] +=x;
				 sds_vec[i] += x*x;

			}
		}
	}

	for (int i=0; i < numberVariablesThisStep; ++i)
	{
		means_vec[i] /=validPaths;
		sds_vec[i]  /=validPaths;
		sds_vec[i] -= means_vec[i]*means_vec[i];
		sds_vec[i] = sqrt(sds_vec[i]);

	}




	for (int p=0; p < totalPaths; ++p)
	{
		if (allValid || validity[p])
		{

			for (int i=0; i < numberVariablesThisStep; ++i)
			{
				workSpace1_vec[i] = (basisFunctionVariables_cube(stepNumber,i,p)-means_vec[i])/sds_vec[i];
			}
			generator.writeData(workSpace1_vec, workSpace2_vec);

			Realv cashValue = deflatedCashFlowValues_matrix(stepNumber,p) + deflatedNextStepValues_vec[p];


#ifdef _DEBUG
			if (fp_isnan(cashValue))
				throw("isnan!");
#endif

			for (int i=0; i < numberDataPts; ++i)
			{
				for (int j=0; j <= i; ++j)
					regressionMatrix(i,j) += workSpace2_vec[i]*workSpace2_vec[j];

				sumProductsWithContinuationValues[i] += cashValue*workSpace2_vec[i];


			}

		}
	}

	for (int i=0; i < numberDataPts; ++i)
	{
		regressionMatrix(i,i) /= validPaths;
		sumProductsWithContinuationValues[i]/=validPaths;

		for (int j=0; j < i; ++j)
		{
			regressionMatrix(i,j) /=validPaths;
			regressionMatrix(j,i) = regressionMatrix(i,j);
		}

	}

//	debugDumpMatrix(MatrixConstFacade<double>(regressionMatrix), "regressionMatrix");
//	debugDumpVector(sumProductsWithContinuationValues, "sumProductsWithContinuationValues");

}
/*
template<class T>
void updateDeflatedNextStepValues_gold(const MatrixFacade<double>& regressionWeights,
									   Realv AndersenShift,
									   const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
									   const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
									   std::vector<Realv>& deflatedNextStepValues_vec, // deflated to current exercise time
									   const MatrixConstFacade<Realv>& deflatedExerciseValue_matrix,
									   int stepNumber,
									   int totalPaths,
									   int numberVariablesThisStep,
									   T& generator,
									   std::vector<Realv>& workSpace1_vec,
									   std::vector<Realv>& workSpace2_vec
									   )
{
	workSpace1_vec.resize(numberVariablesThisStep);
	int numberDataPts= generator.numberDataPoints();
	workSpace2_vec.resize(numberDataPts);


	for (int p=0; p < totalPaths; ++p)
	{
		for (int i=0; i < numberVariablesThisStep; ++i)
			workSpace1_vec[i] = basisFunctionVariables_cube(stepNumber,i,p);
		generator.writeData(workSpace1_vec, workSpace2_vec);

		Realv estContinuation = AndersenShift;

		for (int j=0; j < numberDataPts; ++j)
			estContinuation+= static_cast<Realv>(workSpace2_vec[j]*regressionWeights(stepNumber,j));

		Realv exValue = deflatedExerciseValue_matrix(stepNumber,p);

		if (exValue > estContinuation)
		{ // exercise
			deflatedNextStepValues_vec[p] = exValue;
		}
		else
		{ // don't exercise 
			deflatedNextStepValues_vec[p] += deflatedCashFlowValues_matrix(stepNumber,p);
		}

	}

}
*/
template<class T, class Iter>
void computedNetContinuationValues_gold(const CubeFacade<double>& regressionWeights,
										int depth,
										Realv AndersenShift,
										const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
										//		   const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
										//	   std::vector<Realv>& deflatedNextStepValues_vec, // deflated to current exercise time
										const MatrixConstFacade<Realv>& deflatedExerciseValue_matrix,
										int stepNumber,
										int totalPaths,
										int numberVariablesThisStep,
										const std::vector<int>& isValid,
										const std::vector<Realv>& means_vec,
										const std::vector<Realv>& sds_vec, 
										T& generator,
										std::vector<Realv>& workSpace1_vec,
										std::vector<Realv>& workSpace2_vec,
										Iter it // for output
										)
{
	workSpace1_vec.resize(numberVariablesThisStep);
	int numberDataPts= generator.numberDataPoints();
	workSpace2_vec.resize(numberDataPts);


	for (int p=0; p < totalPaths; ++p, ++it)
	{
		if (isValid[p])
		{
			for (int i=0; i < numberVariablesThisStep; ++i)
				workSpace1_vec[i] = (basisFunctionVariables_cube(stepNumber,i,p)-means_vec[i])/sds_vec[i];
			generator.writeData(workSpace1_vec, workSpace2_vec);

			Realv estContinuation = AndersenShift;

			for (int j=0; j < numberDataPts; ++j)
				estContinuation+= static_cast<Realv>(workSpace2_vec[j]*regressionWeights(stepNumber,depth,j));

			Realv exValue = deflatedExerciseValue_matrix(stepNumber,p);

			Realv netValue = estContinuation-exValue;
			*it = netValue;

		}
	}

}
/*


template<class T>
double SecondPassPriceLSAUsingAggregatedFlows_gold(int numberExerciseDates,
												   const MatrixConstFacade<double>& LS_coefficients_mat,
												   const std::vector<double>& andersenShifts_vec,
												   const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
												   const std::vector<int>& basisVariablesEachStep,
												   int maxBasisVariables,
												   const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, 
												   const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, 
												   const MatrixFacade<Realv>& numeraireValues_matrix,
												   int paths,
												   const std::vector<int>& deflationLocations_vec,
												   std::vector<Realv>& basisVariableWorkspace_vec,
												   std::vector<Realv>& basisFunctionsWorkspace_vec
												   )
{

	basisVariableWorkspace_vec.resize(maxBasisVariables);

	T generator(maxBasisVariables);

	basisFunctionsWorkspace_vec.resize(generator.numberDataPoints());

	std::vector<T> basisGenerators;
	for (int i=0; i < numberExerciseDates; ++i)
		basisGenerators.push_back(T(basisVariablesEachStep[i]));

	Realv result =0.0;

	for (int p=0; p < paths; ++p)
	{
		Realv totalNumerairesThisPath =0.0;
		bool pathNotDone = true;
		for (int e=0; e < numberExerciseDates && pathNotDone ; ++e)
		{
			for (int i=0; i < basisVariablesEachStep[e]; ++i)
				basisVariableWorkspace_vec[i] = basisFunctionVariables_cube(e,i,p);

			basisGenerators[e].writeData(basisVariableWorkspace_vec, basisFunctionsWorkspace_vec);

			double estCondValue = andersenShifts_vec[e];

			for (int j=0; j < basisGenerators[e].numberDataPoints(); ++j)
				estCondValue += basisFunctionsWorkspace_vec[j]*LS_coefficients_mat(e,j);

			double exValue = deflatedExerciseValues_matrix(e,p);
			double cfValue = deflatedCashFlowValues_matrix(e,p);

			if (estCondValue > exValue)
			{
				totalNumerairesThisPath += cfValue/numeraireValues_matrix(deflationLocations_vec[e],p);
			}
			else
			{
				totalNumerairesThisPath += exValue/numeraireValues_matrix(deflationLocations_vec[e],p);
				pathNotDone = false;
			}


		}

		result += totalNumerairesThisPath;


	}

	result/=paths;

	return result;
}



template<class T>
double SecondPassPriceLSAUsingAggregatedFlows_gold(int numberExerciseDates,
												   const MatrixConstFacade<double>& LS_coefficients_mat,
												   const std::vector<double>& andersenShifts_vec,
												   const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
												   const std::vector<int>& basisVariablesEachStep,
												   int maxBasisVariables,
												   const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
												   const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, // deflated to current exercise time
												   const MatrixFacade<Realv>& numeraireValues_matrix,
												   int paths,
												   const std::vector<int>& deflationLocations_vec)
{
	std::vector<Realv> basisVariableWorkspace_vec;
	std::vector<Realv> basisFunctionsWorkspace_vec;

	return SecondPassPriceLSAUsingAggregatedFlows_gold<T>( numberExerciseDates,
		LS_coefficients_mat,
		andersenShifts_vec,
		basisFunctionVariables_cube,
		basisVariablesEachStep,
		maxBasisVariables,
		deflatedCashFlowValues_matrix, // deflated to current exercise time
		deflatedExerciseValues_matrix, // deflated to current exercise time
		numeraireValues_matrix,
		paths,
		deflationLocations_vec,
		basisVariableWorkspace_vec,
		basisFunctionsWorkspace_vec);
}                                       


*/


#endif

