//
//
//                  LS_basis_gold.h
//
//
// (c) Mark Joshi 2011, 2013,2014
// This code is released under the GNU public licence version 3

#ifndef LS_BASIS_GOLD_H
#define LS_BASIS_GOLD_H

#include <gold/MatrixFacade.h>
#include <vector>
#include <gold/math/typedefs_math_gold.h>
#include <gold/LS_basis_examples_gold.h>




template<class T,class D> 
void adjoinBasisVariablesCaller_gold(const std::vector<int>& integerData_vec, // int data specific to the basis variables
		                  const std::vector<D>& floatData_vec, // float data specific to the basis variables
                          const CubeConstFacade<D>&  forwards_cube,
                          const CubeConstFacade<D>&  discountRatios_cube,
                          const MatrixConstFacade<D>&  rate1_matrix,
                          const MatrixConstFacade<D>&  rate2_matrix,
                          const MatrixConstFacade<D>&  rate3_matrix,
                          const int numberPathsPerBatch,
                          const int totalNumberOfPaths, 
                          const int numberPathsPreviouslyDone,
                          const int numberStepsAndRates, 
                          const int numberExerciseTimes,
                          const std::vector<int>&  exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
                          const std::vector<int>&  exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                          CubeFacade<D>&  basisFunctionVariables_cube,// output location
                          int stepsAlreadyDone
                          )
{
   T object(integerData_vec,
		    floatData_vec, 
            numberStepsAndRates);

   int batchLayerSize = numberPathsPerBatch*numberStepsAndRates ;

   int e=0;
   while (exerciseIndices_vec[e] < stepsAlreadyDone)
       ++e;

   for (int i =0; i < numberPathsPerBatch; ++i)
       for (int j=e; j < numberExerciseTimes; ++j)
       {
            int step = exerciseIndices_vec[j];


            D rate1 = rate1_matrix(j,i);
            D rate2 = rate2_matrix(j,i);
            D rate3 = rate3_matrix(j,i);

            object.writeBasisVariableValues(basisFunctionVariables_cube, 
                                            step,
                                            j,
                                            numberStepsAndRates,
                                            rate1,
                                            rate2,
                                            rate3,
                                            i,
                                            numberPathsPreviouslyDone+i,
                                            forwards_cube,
                                            discountRatios_cube,
                                            numberPathsPerBatch,
                                            batchLayerSize);

       }
    
}

// for testing adjoin basis code with the simplest basis functions
template<class D>
void adjoinBasisVariablesCaller_gold(bool useLog,
									 const std::vector<int>& integerData_vec, // int data specific to the basis variables
                                     const std::vector<D>& floatData_vec, // float data specific to the basis variables
                                     const CubeConstFacade<D>&  forwards_cube,
                                     const CubeConstFacade<D>&  discountRatios_cube,
                                     const MatrixConstFacade<D>&  rate1_matrix,
                                     const MatrixConstFacade<D>&  rate2_matrix,
                                     const MatrixConstFacade<D>&  rate3_matrix,
                                     const int numberPathsPerBatch,
                                     const int totalNumberOfPaths, 
                                     const int numberPathsPreviouslyDone,
                                     const int numberStepsAndRates, 
                                     const std::vector<int>&  exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
                                     const std::vector<int>&  exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                                     CubeFacade<D>&  basisFunctionVariables_cube,// output location
                                     int stepsAlreadyDone=0
                                     )
{
	if (useLog)
		adjoinBasisVariablesCaller_gold<basisVariableLog_gold<D>,D>(integerData_vec, // int data specific to the basis variables
		floatData_vec, // float data specific to the basis variables
		forwards_cube,
		discountRatios_cube,
		rate1_matrix,
		rate2_matrix,
		rate3_matrix,
		numberPathsPerBatch,
		totalNumberOfPaths, 
		numberPathsPreviouslyDone,
		numberStepsAndRates, 
		static_cast<int>(exerciseIndices_vec.size()),
		exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
		exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
		basisFunctionVariables_cube// output location
        ,stepsAlreadyDone
		);



	else

		adjoinBasisVariablesCaller_gold<basisVariableExample_gold<D>,D>(integerData_vec, // int data specific to the basis variables
		floatData_vec, // float data specific to the basis variables
		forwards_cube,
		discountRatios_cube,
		rate1_matrix,
		rate2_matrix,
		rate3_matrix,
		numberPathsPerBatch,
		totalNumberOfPaths, 
		numberPathsPreviouslyDone,
		numberStepsAndRates, 
		static_cast<int>(exerciseIndices_vec.size()),
		exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
		exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
		basisFunctionVariables_cube// output location
        ,stepsAlreadyDone
		);

}


void expandBasisFunctions_quadratic_gold(
	int totalNumberOfPaths, 
	const std::vector<Realv>& basisVariables_vec, // input location for basis variables for all steps,
	int exerciseNumber,
	int numberExerciseDates, 
	int maxBasisVariableSize, 
	int variableSizeThisStep,
	bool useCrossTerms,
	std::vector<Realv>& basisFunctions_vec
	);

#endif
