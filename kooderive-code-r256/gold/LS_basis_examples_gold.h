//

//
//
//                  LS_basis_examples_gold.h
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef LS_BASIS_EXAMPLES_GOLD_H
#define LS_BASIS_EXAMPLES_GOLD_H

#include <gold/MatrixFacade.h>
#include <vector>
#include <gold/math/typedefs_math_gold.h>

template<class D>
class basisVariableExample_gold
{
private:

    int numberStepsAndRates_;

public:

    basisVariableExample_gold(	const std::vector<int>& auxDataInt_vec, // 
		                const std::vector<D>& auxDataFloat_vec, 
                        int numberStepsAndRates) : numberStepsAndRates_(numberStepsAndRates)
	{
	}

void writeBasisVariableValues(CubeFacade<D>& output_cube, 
		int numberOfRatesReset,  // the index amongst the rate times 
        int exerciseIndex, // the index amongst the exercise times 
		int numberRates, 
		D rate1, 
		D rate2, 
		D rate3, 
        int inputPathNumber,
        int outputPathNumber,
		const CubeConstFacade<D>& allForwardRates, 
		const CubeConstFacade<D>& allDiscountRatios,
        int pathsPerBatch,
        int batchLayerSize
		);

static int maxVariablesPerStep()
{   
    return 3;
}
static int actualNumberOfVariables(int stepIndex, int aliveIndex, int numberOfRates)
{
	if (aliveIndex+3 <= numberOfRates)
	    return 3;
	if (aliveIndex+2 == numberOfRates)
		return 2;
	if (aliveIndex+1 == numberOfRates)
		return 1;
	return 0; // this is only reached if you have called with silly values,
			  // i.e. the aliveIndex is beyond all the rates
}


};
template<class D>
class basisVariableLog_gold
{
private:

    int numberStepsAndRates_;
	std::vector<D> shifts_;

public:

    basisVariableLog_gold(	const std::vector<int>& auxDataInt_vec, // 
		                const std::vector<D>& auxDataFloat_vec, 
                        int numberStepsAndRates) ;

void writeBasisVariableValues(CubeFacade<D>& output_cube, 
		int numberOfRatesReset,  // the index amongst the rate times 
        int exerciseIndex, // the index amongst the exercise times 
		int numberRates, 
		D rate1, 
		D rate2, 
		D rate3, 
        int inputPathNumber,
        int outputPathNumber,
		const CubeConstFacade<D>& allForwardRates, 
		const CubeConstFacade<D>& allDiscountRatios,
        int pathsPerBatch,
        int batchLayerSize
		);

static int maxVariablesPerStep()
{   
    return 3;
}

static int actualNumberOfVariables(int stepIndex, int aliveIndex, int numberOfRates)
{
if (aliveIndex+3 <= numberOfRates)
	    return 3;
	if (aliveIndex+2 == numberOfRates)
		return 2;
	if (aliveIndex+1 == numberOfRates)
		return 1;
	return 0; // this is only reached if you have called with silly values,
			  // i.e. the aliveIndex is beyond all the rates
}

};


template<class D>
void basisVariableExample_gold<D>::writeBasisVariableValues(CubeFacade<D>& output_cube, 
		int numberOfRatesReset,  // the index amongst the rate times 
        int exerciseIndex, // the index amongst the exercise times 
		int numberRates, 
		D rate1, 
		D rate2, 
		D rate3, 
        int inputPathNumber,
        int outputPathNumber,
		const CubeConstFacade<D>& allForwardRates, 
		const CubeConstFacade<D>& allDiscountRatios,
        int pathsPerBatch,
        int batchLayerSize
		)
{
    D variable1 = rate1;
    D variable2 = rate2;
    D variable3 = rate3;
    output_cube(exerciseIndex,0,outputPathNumber)=variable1;
    output_cube(exerciseIndex,1,outputPathNumber)=variable2;
    output_cube(exerciseIndex,2,outputPathNumber)=variable3;

}

template<class D>
basisVariableLog_gold<D>::basisVariableLog_gold(	const std::vector<int>& auxDataInt_vec, // 
		                const std::vector<D>& auxData_vec, 
                        int numberStepsAndRates) : numberStepsAndRates_(numberStepsAndRates), shifts_(auxData_vec)
{
	if ( shifts_.size() < 3)
		GenerateError("shifts inputs too short in basisVariableLog_gold ");


}

template<class D>
void basisVariableLog_gold<D>::writeBasisVariableValues(CubeFacade<D>& output_cube, 
		int numberOfRatesReset,  // the index amongst the rate times 
        int exerciseIndex, // the index amongst the exercise times 
		int numberRates, 
		D rate1, 
		D rate2, 
		D rate3, 
        int inputPathNumber,
        int outputPathNumber,
		const CubeConstFacade<D>& allForwardRates, 
		const CubeConstFacade<D>& allDiscountRatios,
        int pathsPerBatch,
        int batchLayerSize
		)
{
	D variable1 = log(rate1+shifts_[0]);
    D variable2 = log(rate2+shifts_[1]);
    D variable3 = log(rate3+shifts_[2]);
    output_cube(exerciseIndex,0,outputPathNumber)=variable1;
    output_cube(exerciseIndex,1,outputPathNumber)=variable2;
    output_cube(exerciseIndex,2,outputPathNumber)=variable3;

}

#endif
