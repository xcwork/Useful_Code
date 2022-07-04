/*

#include "cula_test.h"
#include <LS_main_cula.h>

#include <gold/math/typedefs_math_gold.h>
#include <gold/math/basic_matrix_gold.h>
#include <gold/math/fp_utilities.h>
#include <gold/math/pseudoSquareRoot.h>
#include <gold/math/basic_matrix_gold.h>
#include <Utilities.h>
#include <cmath>
#include <iostream>
#include <cutil_inline.h>
#include <cutil.h>

#include <cula_lapack_device.h>


int Test_cula_solve_square(bool verbose)
{
	// solve A^t x = y and over write y to put x in y


	//	void culaSolve(int A_columns, int A_rows, float* A_global, float* y_global)

	std::cout << "Entering cula test.\n";
	cudaSetDevice(cutGetMaxGflopsDeviceId());
	cudaThreadSynchronize();
	culaInitialize();

	int result = 20;
	Real_d tolerance= 1e-4;

	for (int k=1; k < 3; ++k)
	{
		for (int r = 1; r< 11; ++r)
		{
			std::vector<float> matrix1_vec(r*r);
			MatrixFacade<float> Matrix1_mat(&matrix1_vec[0],r,r);

			std::vector<float> matrix2_vec(r*r);
			MatrixFacade<float> Matrix2_mat(&matrix2_vec[0],r,r);

			std::vector<float> matrix3_vec(r*r);
			MatrixFacade<float> Matrix3_mat(&matrix3_vec[0],r,r);

			std::vector<float> matrix4_vec(r*r);
			MatrixFacade<float> Matrix4_mat(&matrix4_vec[0],r,r);


			for (int i=0; i < r; ++i)
				for (int j=0; j <= i; ++j)
				{
					Matrix1_mat(i,j) = i+j+1.0f+k;
					Matrix2_mat(j,i) = i+j+3.0f+k;
				}

				MatrixMatrixProduct<float>(Matrix1_mat,Matrix2_mat,Matrix3_mat);
				MatrixTranspose<float>(Matrix3_mat,Matrix4_mat);

				std::vector<float> target(r);

				for (int i=0; i < r; ++i)
					target[i] = i+2.5f+k;


				thrust::device_vector<float> matrix4_gpu(deviceVecFromStlVec(matrix4_vec));
				thrust::device_vector<float> target_gpu(deviceVecFromStlVec(target));

				culaSolve(Matrix4_mat.columns(),
					Matrix4_mat.rows(),
					thrust::raw_pointer_cast(&  matrix4_gpu[0]),
					thrust::raw_pointer_cast(&  target_gpu[0]));


				std::vector<float> solution(stlVecFromDevVec(target_gpu));

					SVDSquareMatrix svd(r);

				svd.SVDSolveBoost( Matrix3_mat, 
				target, 
				solution);



				std::vector<float> target_comp(r);

				MatrixArrayProduct<float>(Matrix3_mat,solution,target_comp);



				Real_d error=0.0;

				for (int j=0; j < r; ++j)
					error += fabs(target[j]-target_comp[j]);


				if (error > tolerance || fp_isnan(error))
					std::cout << "cula sgels test " << r << " failed with error "<< error<< "\n";

				else --result;







		}
	}
	culaShutdown();
	cudaThreadExit();

	if (result ==0)
		std::cout << "cula square test passed.\n";

	return result;

}


int Test_cula_solve_rectangular(bool verbose)
{

	int result =1;
	

	std::cout << "Entering cula rectangle test.\n";
	cudaSetDevice(cutGetMaxGflopsDeviceId());
	cudaThreadSynchronize();
	culaInitialize();
	{

		Real_d tolerance= 1e-4;

		int dataPoints = 1000;
		int basisFunctions = 10;

		std::vector<float> Y_vec(dataPoints*basisFunctions);
		MatrixFacade<float> Y_mat(&Y_vec[0],basisFunctions,dataPoints);

		std::vector<float> Y_T_vec(dataPoints*basisFunctions);
		MatrixFacade<float> Y_T_mat(&Y_vec[0],dataPoints,basisFunctions);


		for (int i=0; i < dataPoints; ++i)
			for (int k=0; k < basisFunctions; ++k)
				Y_mat(k,i) = cos(i*k+0.0f);

		MatrixTranspose<float>(Y_mat,Y_T_mat);

		std::vector<float> alpha_original_vec(basisFunctions);

		for (int i=0; i < basisFunctions; ++i)
			alpha_original_vec[i] = i/10.0f;

		std::vector<float> target_vec(dataPoints);

		MatrixArrayProduct<float>(Y_T_mat,alpha_original_vec,target_vec);

		thrust::device_vector<float> Y_gpu(deviceVecFromStlVec(Y_vec));
		thrust::device_vector<float> target_gpu(deviceVecFromStlVec(target_vec));

		culaSolve(Y_mat.columns(),
			Y_mat.rows(),
			thrust::raw_pointer_cast(&  Y_gpu[0]),
			thrust::raw_pointer_cast(&  target_gpu[0]),'T');


		std::vector<float> solution(stlVecFromDevVec(target_gpu));
		solution.resize(basisFunctions);





		Real_d error=0.0;

		for (int j=0; j < basisFunctions; ++j)
			error += fabs(alpha_original_vec[j]-solution[j]);


		if (error > tolerance || fp_isnan(error))
			std::cout << "cula rectabgular sgels test  failed with error "<< error<< "\n";
		else --result;





	}

	culaShutdown();
	cudaThreadExit();

	if (result ==0)
		std::cout << "cula test passed.\n";

	return result;

}
int Test_cula_solve(bool verbose)
{
	return Test_cula_solve_rectangular(verbose);//+Test_cula_solve_square(verbose);

}
*/
