// Schur_gold.cpp
// (c) Mark Joshi 2013
// released under GPL v 3.0
// derived from code with license below


/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

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

#include <gold/math/Schur_gold.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

SymmetricSchurDecomposition::SymmetricSchurDecomposition(const MatrixConstFacade<Real_d> & s)
: diagonal_(s.rows()), 
eigenVectors_(s.rows(), s.columns(), 0.0) 
{

	if (s.rows()!=s.columns())
		GenerateError("input matrix must be square");

	int size = s.rows();
	for (int q=0; q<size; q++) {
		diagonal_[q] = s[q][q];
		eigenVectors_[q][q] = 1.0;
	}
	Matrix_gold<Real_d> ss = s;

	std::vector<Real_d> tmpDiag(diagonal_.begin(), diagonal_.end());
	std::vector<Real_d> tmpAccumulate(size, 0.0);
	Real_d threshold, epsPrec = 1e-15;
	bool keeplooping = true;
	int maxIterations = 100, ite = 1;
	do {
		//main loop
		Real_d sum = 0;
		for (int a=0; a<size-1; a++) 
		{
			for (int b=a+1; b<size; b++) 
			{
				sum += std::fabs(ss[a][b]);
			}
		}

		if (sum==0) 
		{
			keeplooping = false;
		} 
		else 
		{
			/* To speed up computation a threshold is introduced to
			make sure it is worthy to perform the Jacobi rotation
			*/
			if (ite<5)
				threshold = 0.2*sum/(size*size);
			else       
				threshold = 0.0;

			int j, k, l;
			for (j=0; j<size-1; j++) 
			{
				for (k=j+1; k<size; k++) 
				{
					Real_d sine, rho, cosin, heig, tang, beta;
					Real_d smll = std::fabs(ss[j][k]);
					if(ite> 5 &&
						smll<epsPrec*std::fabs(diagonal_[j]) &&
						smll<epsPrec*std::fabs(diagonal_[k])) 
					{
						ss[j][k] = 0;
					} 
					else if (std::fabs(ss[j][k])>threshold) 
					{
						heig = diagonal_[k]-diagonal_[j];
						if (smll<epsPrec*std::fabs(heig)) {
							tang = ss[j][k]/heig;
						} else {
							beta = 0.5*heig/ss[j][k];
							tang = 1.0/(std::fabs(beta)+
								std::sqrt(1+beta*beta));
							if (beta<0)
								tang = -tang;
						}
						cosin = 1/std::sqrt(1+tang*tang);
						sine = tang*cosin;
						rho = sine/(1+cosin);
						heig = tang*ss[j][k];
						tmpAccumulate[j] -= heig;
						tmpAccumulate[k] += heig;
						diagonal_[j] -= heig;
						diagonal_[k] += heig;
						ss[j][k] = 0.0;
						for (l=0; l+1<=j; l++)
							jacobiRotate_(ss, rho, sine, l, j, l, k);
						for (l=j+1; l<=k-1; l++)
							jacobiRotate_(ss, rho, sine, j, l, l, k);
						for (l=k+1; l<size; l++)
							jacobiRotate_(ss, rho, sine, j, l, k, l);
						for (l=0;   l<size; l++)
							jacobiRotate_(eigenVectors_,
							rho, sine, l, j, l, k);
					}
				}
			}
			for (k=0; k<size; k++) {
				tmpDiag[k] += tmpAccumulate[k];
				diagonal_[k] = tmpDiag[k];
				tmpAccumulate[k] = 0.0;
			}
		}
	} while (++ite<=maxIterations && keeplooping);

	if (ite > maxIterations)
		GenerateError(
		"Too many iterations ");

	// sort (eigenvalues, eigenvectors)
	std::vector<std::pair<Real_d, std::vector<Real_d> > > temp(size);
	std::vector<Real_d> eigenvector(size);
	int row, col;
	for (col=0; col<size; col++) 
	{
		for (int row=0; row<size; row++)
			eigenvector[row] = eigenVectors_(row,col);

		temp[col] = std::make_pair(diagonal_[col], eigenvector);
	}

	std::sort(temp.begin(), temp.end(),
		std::greater<std::pair<Real_d, std::vector<Real_d> > >());

	Real_d maxEv = temp[0].first;

	for (col=0; col<size; col++) 
	{
		// check for round-off errors
		diagonal_[col] =
			(std::fabs(temp[col].first/maxEv)<1e-16 ? 0.0 :
			temp[col].first);

		Real_d sign = 1.0;

		if (temp[col].second[0]<0.0)
			sign = -1.0;

		for (row=0; row<size; row++) 
		{
			eigenVectors_[row][col] = sign * temp[col].second[row];
		}
	}
}
