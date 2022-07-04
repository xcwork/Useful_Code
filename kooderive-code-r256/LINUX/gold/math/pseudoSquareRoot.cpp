
////                     pseudoSquareRoot.h

//  Copyright (C) 2012 Mark Joshi

// Release under GNU public licence version 3
/*
Copyright (C) 2003, 2004, 2007 Ferdinando Ametrano
Copyright (C) 2006 Yiping Chen
Copyright (C) 2007 Neil Firth

This file is part of QuantLib, a free-software/open-source library
for financial quantitative analysts and developers - http://quantlib.org/

QuantLib is free software: you can redistribute it and/or modify it
under the terms of the QuantLib license.  You should have received a
copy of the license along with this program; if not, please email
<quantlib-dev@lists.sf.net>. The license is also available online at
<http://quantlib.org/license.shtml>.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the license for more details.
*/
#include <gold/math/pseudoSquareRoot.h>
#include <gold/math/typedefs_math_gold.h>

#include <gold/MatrixFacade.h>

#include <cmath>
#include <gold/math/basic_matrix_gold.h>
#include <gold/math/svd_gold.h>
#include <gold/math/Schur_gold.h>

void pseudoSqrtSpectral(
						const MatrixConstFacade<Real_d>& matrix,
						MatrixFacade<Real_d>& result)
{
	size_t size = matrix.rows();

	Real_d tolerance = 1E-12;

	std::vector<Real_d> singular(size);
	std::vector<Real_d> U(size*size);
	std::vector<Real_d> V(size*size);
	MatrixFacade<Real_d> U_mat(U,size,size);
	MatrixFacade<Real_d> V_mat(V,size,size);

	SVDSquareMatrix solver(size);

	solver.GetDecomp( matrix,  singular,
		U_mat,
		V_mat,
		tolerance);

	std::vector<Real_d> diagonal_vec(size* size, 0.0);

	MatrixFacade<Real_d> diagonal_mat(diagonal_vec,size,size);


	// negative eigenvalues set to zero
	for (size_t i=0; i<size; i++)
		diagonal_mat(i,i) =
		std::sqrt(std::max<Real_d>(singular[i], 0.0));

	MatrixMatrixProduct<Real_d>(U_mat , diagonal_mat,result);




}




void pseudoSqrtSchur(const MatrixConstFacade<Real_d>& input, MatrixFacade<Real_d>& output, int factors, bool normalise )
{
	SymmetricSchurDecomposition decomp(input);

	for (int f=0; f < factors; ++f)
	{
		Real_d lambda = decomp.eigenvalues()[f];
		Real_d rootLambda = lambda > 0.0 ? sqrt(lambda) : 0.0;

		for (int i=0; i < input.rows(); ++i)
			output(i,f) = decomp.eigenvectors()(i,f) * rootLambda;
	}

	if (normalise)
		for (int i=0; i< input.rows(); ++i) 
		{

			Real_d norm = 0.0;
			for (int j=0; j<factors; ++j)
				norm += output[i][j]*output[i][j];

			if (norm>0.0) 
		 {
			 Real_d normAdj = std::sqrt(input[i][i]/norm);

			 for (int j=0; j<factors; ++j)
				 output[i][j] *= normAdj;
			}

		}




}
