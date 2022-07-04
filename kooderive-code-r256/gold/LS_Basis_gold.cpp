//
//
//                  LS_basis_gold.cpp
//
//
// (c) Mark Joshi 2011,2013
// This code is released under the GNU public licence version 3

#include <gold/LS_basis_gold.h>
#include <gold/LS_regression.h>
#include <gold/MatrixFacade.h>


/*
This really needs to be rewritten to make polynomial type a template parameter
or virtual function. 
*/
void expandBasisFunctions_quadratic_gold(
	int totalNumberOfPaths, 
	const std::vector<Realv>& basisVariables_vec, // input location for basis variables for all steps,
	int exerciseNumber,
	int numberExerciseDates, 
	int maxBasisVariableSize, 
	int variableSizeThisStep,
	bool useCrossTerms,
	std::vector<Realv>& basisFunctions_vec
	)

{
	CubeConstFacade<Realv> basis_cube(basisVariables_vec,numberExerciseDates,maxBasisVariableSize,totalNumberOfPaths);


	quadraticPolynomialGenerator poly(variableSizeThisStep);
	quadraticPolynomialCrossGenerator polyCross(variableSizeThisStep);

	int outSize = useCrossTerms ? polyCross.numberDataPoints() : poly.numberDataPoints();
	basisFunctions_vec.resize(outSize*totalNumberOfPaths);
	MatrixFacade<Realv>  functions_output_mat(basisFunctions_vec,outSize,totalNumberOfPaths);

	std::vector<Realv> inputs(variableSizeThisStep);
	std::vector<Realv> outputs(outSize);


	for (int p=0; p < totalNumberOfPaths; ++p)
	{
		for (int i=0; i < variableSizeThisStep; ++i)
			inputs[i] = basis_cube(exerciseNumber,i,p);

		if (useCrossTerms)
			polyCross.writeData(inputs,outputs);
		else
			poly.writeData(inputs,outputs);
		
		for (int i=0; i < outSize; ++i)
			functions_output_mat(i,p) = outputs[i];

	}

}

