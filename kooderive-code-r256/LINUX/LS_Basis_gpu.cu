//
//
//                  LS_basis_gpu.cu
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3
#include <LS_Basis_gpu.h>


extern"C"
void adjoinBasisVariablesCaller_gpu(bool useLog,
									int* integerData_dev, // to access texture containing data specific to the basis variables
		                  float* floatData_dev, // to access texture containing data specific to the basis variables
                          float* forwards_dev,
                          float* discountRatios_dev,
                          float* rate1_dev,
                          float* rate2_dev,
                          float* rate3_dev,
                          int numberPathsPerBatch,
                          int totalNumberOfPaths, 
                          int numberPathsPreviouslyDone,
                          int numberStepsAndRates, 
                          int numberExerciseTimes,
                          int* exerciseIndices_dev, // the indices of the exercise times amongst the evolution times
                          int* exerciseIndicators_dev, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                          float* basisFunctionVariables_dev // output location
                          )
{
	if (useLog)
		    adjoinBasisVariables<basisVariableLog_gpu>(integerData_dev, // to access texture containing data specific to the basis variables
		                  floatData_dev, // to access texture containing data specific to the basis variables
                          forwards_dev,
                          discountRatios_dev,
                          rate1_dev,
                          rate2_dev,
                          rate3_dev,
                          numberPathsPerBatch,
                          totalNumberOfPaths, 
                          numberPathsPreviouslyDone,
                          numberStepsAndRates, 
                          numberExerciseTimes,
                          exerciseIndices_dev, // the indices of the exercise times amongst the evolution times
                          exerciseIndicators_dev, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                          basisFunctionVariables_dev // output location
                          );
	else
	    adjoinBasisVariables<basisVariableExample>(integerData_dev, // to access texture containing data specific to the basis variables
		                  floatData_dev, // to access texture containing data specific to the basis variables
                          forwards_dev,
                          discountRatios_dev,
                          rate1_dev,
                          rate2_dev,
                          rate3_dev,
                          numberPathsPerBatch,
                          totalNumberOfPaths, 
                          numberPathsPreviouslyDone,
                          numberStepsAndRates, 
                          numberExerciseTimes,
                          exerciseIndices_dev, // the indices of the exercise times amongst the evolution times
                          exerciseIndicators_dev, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                          basisFunctionVariables_dev // output location
                          );
}

