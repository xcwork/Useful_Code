// Schur_gold.h
// (c) Mark Joshi 2013
// released under GPL v 3.0

// derived from code with license below
/*
Copyright (C) 2003 Ferdinando Ametrano
Copyright (C) 2000, 2001, 2002, 2003 RiskMap srl

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

#ifndef SCHUR_GOLD_H
#define SCHUR_GOLD_H

#include <gold/math/basic_matrix_gold.h>
#include <gold/math/typedefs_math_gold.h>

//! symmetric threshold Jacobi algorithm.
/*! Given a real symmetric matrix S, the Schur decomposition
finds the eigenvalues and eigenvectors of S. If D is the
diagonal matrix formed by the eigenvalues and U the
unitarian matrix of the eigenvectors we can write the
Schur decomposition as
\f[ S = U \cdot D \cdot U^T \, ,\f]
where \f$ \cdot \f$ is the standard matrix product
and  \f$ ^T  \f$ is the transpose operator.
This class implements the Schur decomposition using the
symmetric threshold Jacobi algorithm. For details on the
different Jacobi transfomations see "Matrix computation,"
second edition, by Golub and Van Loan,
The Johns Hopkins University Press

\test the correctness of the returned values is tested by
checking their properties.
*/
class SymmetricSchurDecomposition 
{
public:
	/*! \pre s must be symmetric */
	SymmetricSchurDecomposition(const MatrixConstFacade<Real_d>& s);

	const std::vector<Real_d>& eigenvalues() const 
	{ 
		return diagonal_; 
	}

	const Matrix_gold<Real_d>& eigenvectors() const 
	{ 
		return eigenVectors_; 
	}

private:
	std::vector<Real_d> diagonal_;
	Matrix_gold<Real_d> eigenVectors_;
	void jacobiRotate_(Matrix_gold<Real_d>& m, Real_d rot, Real_d dil,
		int j1, int k1, int j2, int k2) const;
};


// inline definitions

//! This routines implements the Jacobi, a.k.a. Givens, rotation
inline void SymmetricSchurDecomposition::jacobiRotate_(
	Matrix_gold<Real_d> &m, 
	Real_d rot, 
	Real_d dil, 
	int j1,
	int k1, 
	int j2, 
	int k2) const 
{
	Real_d x1, x2;
	x1 = m[j1][k1];
	x2 = m[j2][k2];
	m[j1][k1] = x1 - dil*(x2 + x1*rot);
	m[j2][k2] = x2 + dil*(x1 - x2*rot);
}



#endif


