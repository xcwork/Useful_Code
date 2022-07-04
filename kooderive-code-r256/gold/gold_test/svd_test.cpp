
#include <gold/math/svd_gold.h>
#include <gold/math/typedefs_math_gold.h>
#include <gold/math/basic_matrix_gold.h>
#include <gold/math/fp_utilities.h>
#include <gold/math/pseudosquareRoot.h>
#include <gold/math/basic_matrix_gold.h>
#include <cmath>
#include <iostream>

int svdtest(bool verbose)
{
	std::cout << "Entering SVD test.\n";
	int result = 20;
	Real_d tolerance= 1e-8;

	for (int k=1; k < 3; ++k)
	{
		for (int r = 1; r< 11; ++r)
		{
			std::vector<Real_d> matrix1_vec(r*r);
			MatrixFacade<Real_d> Matrix1_mat(&matrix1_vec[0],r,r);

			std::vector<Real_d> matrix2_vec(r*r);
			MatrixFacade<Real_d> Matrix2_mat(&matrix2_vec[0],r,r);

			std::vector<Real_d> matrix3_vec(r*r);
			MatrixFacade<Real_d> Matrix3_mat(&matrix3_vec[0],r,r);

			for (int i=0; i < r; ++i)
				for (int j=0; j <= i; ++j)
				{
					Matrix1_mat(i,j) = i+j+1+k;
					Matrix2_mat(j,i) = i+j+3+k;
				}

				MatrixMatrixProduct<Real_d>(Matrix1_mat,Matrix2_mat,Matrix3_mat);

				std::vector<Real_d> target(r);

				for (int i=0; i < r; ++i)
					target[i] = i+2.5+k;

				std::vector<Real_d> solution(r);

				SVDSquareMatrix svd(r);

				svd.SVDSolve( Matrix3_mat, 
					target, 
					solution,tolerance);

				std::vector<Real_d> target_comp(r);

				MatrixArrayProduct<Real_d>(Matrix3_mat,solution,target_comp);

				Real_d error=0.0;

				for (int j=0; j < r; ++j)
					error += fabs(target[j]-target_comp[j]);


				if (error > tolerance || fp_isnan(error))
					std::cout << "svd test " << r << " failed\n";

				else --result;







		}
	}

	if (result ==0)
		std::cout << "Svd test passed.\n";

	return result;

}

int pseudoRootTest(bool verbose)
{
	int result=2;

	if (verbose)
		std::cout << "entering pseudo root test\n";

	int n =30;
	Real_d rho =0.5;

	std::vector<Real_d> data_vec(n*n,rho);
	{
		MatrixFacade<Real_d> data_mat(data_vec,n,n);

		for (int i=0; i < n;++i)
			data_mat(i,i) = 1.0;

		std::vector<Real_d> out_vec(n*n);
		MatrixFacade<Real_d> out_mat(out_vec,n,n);


		pseudoSqrtSpectral( data_mat,
			out_mat);
		{

			Matrix_gold<Real_d> test_mat(n,n,0.0);


			for (int i=0; i < n; ++i)
				for (int j=0; j < n; ++j)
				{
					Real_d res =0.0;


					for (int f=0; f < n; ++f)
						res+= out_mat(i,f)*out_mat(j,f);

					test_mat(i,j)=res;

				}

				Real_d tolerance = 1E-12;

				int errs=0;

				for (int i=0; i < n; ++i)
					for (int j=0; j < n; ++j)
						if (fabs(data_mat(i,j)-test_mat(i,j)) > tolerance)
							++errs;



				if (errs >0)
					std::cout << "pseudo root test 1 failed with "<< errs << "\n";
				else
				{
					std::cout << "pseudo root test  1 passed";
					--result;
				}
		}

	}

	{
		MatrixFacade<Real_d> data_mat(data_vec,n,n);

		for (int i=0; i < n;++i)
			data_mat(i,i) = 1.0;

		std::vector<Real_d> out_vec(n*n);
		MatrixFacade<Real_d> out_mat(out_vec,n,n);


		pseudoSqrtSchur( data_mat,
			out_mat,n,false);
		{

			std::vector<Real_d> test_vec(n*n);
			MatrixFacade<Real_d> test_mat(test_vec,n,n);


			for (int i=0; i < n; ++i)
				for (int j=0; j < n; ++j)
				{
					Real_d res =0.0;


					for (int f=0; f < n; ++f)
						res+= out_mat(i,f)*out_mat(j,f);

					test_mat(i,j)=res;

				}

				Real_d tolerance = 1E-12;

				int errs=0;

				for (int i=0; i < n; ++i)
					for (int j=0; j < n; ++j)
						if (fabs(data_mat(i,j)-test_mat(i,j)) > tolerance)
							++errs;



				if (errs >0)
					std::cout << "pseudo root test 2 failed with "<< errs << "\n";
				else
				{
					std::cout << "pseudo root test 2 passed";
					--result;
				}
		}

	}




	return result;
}
