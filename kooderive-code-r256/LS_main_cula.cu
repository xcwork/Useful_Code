#include <LS_main_cula.h>
//#include <cula_lapack_device.h>

#include <gold/Timers.h>


void debugDumpAATAndAx(thrust::device_vector<float>& A,thrust::device_vector<float>& target, int rows, int columns,std::string identifier)
{
#ifdef _DEBUG

	std::vector<float> A_vec(stlVecFromDevVec(A));
	std::vector<float> x_vec(stlVecFromDevVec(target));
	
	MatrixConstFacade<float> A_mat(A_vec,rows,columns);
	Matrix_gold<double> AAT_mat(rows,rows,0.0);

	std::vector<double> Ax(rows,0.0);

	for (int i=0; i < rows; ++i)
	{
		for (int j=0; j <= i; ++j)
		{
			double x=0.0;
			for (int f=0; f < columns; ++f)
				x += A_mat(i,f)*A_mat(j,f);

			AAT_mat(i,j) = x;
			AAT_mat(j,i) = AAT_mat(i,j);

		}

		for (int k=0; k < columns; ++k)
			Ax[i] += A_mat(i,k)*target[k];
	}



	debugDumpMatrix(AAT_mat.ConstFacade(), identifier.c_str());

	debugDumpVector(Ax,"Ax");
#endif


}


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
														bool useCrossTerms)
{
if (useCrossTerms)
return generateRegressionCoefficientsViaLS_gpu_cula<quadraticPolynomialCrossDevice>( numberExerciseDates,		
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
        paths
        );

else
	return generateRegressionCoefficientsViaLS_gpu_cula<quadraticPolynomialDevice>( numberExerciseDates,		
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
		paths
        );

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
										//				 const std::vector<int>& deflation_locations_vec,
														 int paths,
														 bool useCrossTerms,
														 int numberOfExtraRegressions,
														 RegressionSelector& selector,
														 int minPathsLeft)
{
if (useCrossTerms)
return generateMultiRegressionCoefficientsViaLS_gpu_cula<quadraticPolynomialCrossDevice>(  numberExerciseDates,
														 basisVariableInt_data_device,
														 basisVariableFloat_data_device,
														 coefficients_device, // the LS coefficients are placed here 
														 coefficients_vec, // the LS coefficients are also placed here 
														 lowercuts_device, // the lower cut points for the regressions 
														 uppercuts_device, // the upper cut points for the regressions 
														 basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
														 basisVariablesEachStep_device, //vector of the number of basis variables for each step
														 basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
														 maxBasisVariables,
														 deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
														 deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
														 numeraireValues_device, // numeraire vals for each, matrix of times and paths 
												//		 deflation_locations_vec,
														 paths,
														 numberOfExtraRegressions,
														 selector,
														 minPathsLeft
        );

else
	return generateMultiRegressionCoefficientsViaLS_gpu_cula<quadraticPolynomialDevice>(numberExerciseDates,
														 basisVariableInt_data_device,
														 basisVariableFloat_data_device,
														 coefficients_device, // the LS coefficients are placed here 
														 coefficients_vec, // the LS coefficients are also placed here 
														 lowercuts_device, // the lower cut points for the regressions 
														 uppercuts_device, // the upper cut points for the regressions 
														 basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
														 basisVariablesEachStep_device, //vector of the number of basis variables for each step
														 basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
														 maxBasisVariables,
														 deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
														 deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
														 numeraireValues_device, // numeraire vals for each, matrix of times and paths 
												//		 deflation_locations_vec,
														 paths,
														 numberOfExtraRegressions,
														 selector,
														 minPathsLeft
        );

}


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
														 LinearSolverBaseMain& solver)
{
if (useCrossTerms)
return generateMultiRegressionCoefficientsViaLS_gpu_flexisolve<quadraticPolynomialCrossDevice>(  numberExerciseDates,
														 basisVariableInt_data_device,
														 basisVariableFloat_data_device,
														 coefficients_device, // the LS coefficients are placed here 
														 coefficients_vec, // the LS coefficients are also placed here 
														 lowercuts_device, // the lower cut points for the regressions 
														 uppercuts_device, // the upper cut points for the regressions 
														 basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
														 basisVariablesEachStep_device, //vector of the number of basis variables for each step
														 basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
														 maxBasisVariables,
														 deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
														 deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
														 numeraireValues_device, // numeraire vals for each, matrix of times and paths 
												//		 deflation_locations_vec,
														 paths,
														 numberOfExtraRegressions,
														 selector,
														 minPathsLeft, solver
        );

else
	return generateMultiRegressionCoefficientsViaLS_gpu_flexisolve<quadraticPolynomialDevice>(numberExerciseDates,
														 basisVariableInt_data_device,
														 basisVariableFloat_data_device,
														 coefficients_device, // the LS coefficients are placed here 
														 coefficients_vec, // the LS coefficients are also placed here 
														 lowercuts_device, // the lower cut points for the regressions 
														 uppercuts_device, // the upper cut points for the regressions 
														 basisFunctionVariables_device, // cube of preextracted values for variables of basis funcions
														 basisVariablesEachStep_device, //vector of the number of basis variables for each step
														 basisVariablesEachStep_vec, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
														 maxBasisVariables,
														 deflatedCashFlowValues_device, // deflated to current exercise time, matrix of times and paths 
														 deflatedExerciseValues_device, // deflated to current exercise time, matrix of times and paths 
														 numeraireValues_device, // numeraire vals for each, matrix of times and paths 
												//		 deflation_locations_vec,
														 paths,
														 numberOfExtraRegressions,
														 selector,
														 minPathsLeft, solver
        );

}
