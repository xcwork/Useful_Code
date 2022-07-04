//
//											volstructs_gold.cpp
//
//

// (c) Mark Joshi 2013
// released under GPL v 3.0

// some code derived from QuantLib which carries this licence:
/*

 Copyright (C) 2006, 2007 Ferdinando Ametrano
 Copyright (C) 2006 Cristina Duminuco
 Copyright (C) 2005, 2006 Klaus Spanderen
 Copyright (C) 2007 Giorgio Facchinetti

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



#include <gold/volstructs_gold.h>
#include <cmath>
#include <gold/math/pseudoSquareRoot.h>
#include <gold/math/basic_matrix_gold.h>

Cube_gold<double> FlatVolPseudoRoots(const std::vector<double>& rateTimes,
									 const std::vector<double>& evolutionTimes,
									 const CubeConstFacade<double>& correlationPseudoRoots,
									 const std::vector<double>& vols)
{

	if (evolutionTimes.size() != correlationPseudoRoots.numberLayers())
		GenerateError("evolutionTimes correlationPseudoRoots size mismatch in FlatVolPseudoRoots");

	if (rateTimes.size() != correlationPseudoRoots.numberRows()+1)
		GenerateError("rateTimes correlationPseudoRoots size mismatch in FlatVolPseudoRoots");

	if (vols.size() != correlationPseudoRoots.numberRows())
		GenerateError("vols correlationPseudoRoots size mismatch in FlatVolPseudoRoots");

	int numberRates = correlationPseudoRoots.numberRows();
	int factors = correlationPseudoRoots.numberColumns();
	int steps =  correlationPseudoRoots.numberLayers();

	Cube_gold<double> pseudoRoots(steps,numberRates,factors,0.0);


	double startTime =0.0;

	for (int step=0; step < steps; ++step)
	{
		for (int r=0; r < numberRates; ++r)
		{
			double timeMoving = std::min(evolutionTimes[r],rateTimes[r])-std::min(startTime,rateTimes[r]);
			double sd = vols[r]*sqrt(timeMoving);

			for (int f=0; f < factors; ++f)
			{	
				double arf= sd*correlationPseudoRoots(step,r,f);
				pseudoRoots(step,r,f) = arf;
			}
		}

		startTime = evolutionTimes[step];
	}

	return pseudoRoots;

}
Cube_gold<double> ExponentialLongCorrelationPseudoRoots(const std::vector<double>& rateTimes, 
														int numberSteps,
														double beta,
														double L,
														int factors)
{
	int numberRates = rateTimes.size()+(-1);
	Cube_gold<double> pseudoRoots(numberSteps,numberRates,factors,0.0);

	Matrix_gold<double> correlationMatrix(numberRates,numberRates,0.0);
	for (int i=0; i < numberRates; ++i)
	{
		correlationMatrix(i,i) = 1.0;

		for (int j=0; j < i; ++j)
		{
			double rho = exp(-beta*fabs(rateTimes[i]-rateTimes[j]));
			correlationMatrix(i,j) = correlationMatrix(j,i) = L + (1-L)*rho;
		}

	}

	Matrix_gold<double> correlationPseudoMatrix(numberRates,factors,0.0);

	pseudoSqrtSchur(correlationMatrix, correlationPseudoMatrix.Facade(),  factors, true );


	for (int i=0; i < numberSteps; ++i)
		for (int j=0; j < numberRates; ++j)
			for (int f=0; f < factors; ++f)
				pseudoRoots(i,j,f) = correlationPseudoMatrix(j,f);

	return pseudoRoots;

}
Cube_gold<double> FlatVolPseudoRootsExpCorr(const std::vector<double>& rateTimes,
											const std::vector<double>& evolutionTimes,
											double beta,
											double L,
											int factors,
											const std::vector<double>& vols)
{
	Cube_gold<double> pseudoCorrs(ExponentialLongCorrelationPseudoRoots(rateTimes,evolutionTimes.size(),beta,L,factors));

	return  FlatVolPseudoRoots(rateTimes,
		evolutionTimes,
		pseudoCorrs,
		vols);
}


Cube_gold<float> FlatVolPseudoRootsFloat(const std::vector<float>& rateTimes,
										 const std::vector<float>& evolutionTimes,
										 float beta,
										 float L,
										 int factors,
										 const std::vector<float>& vols)
{

	std::vector<double> rateTimesDouble(rateTimes.begin(),rateTimes.end());
	std::vector<double> evolutionTimesDouble(evolutionTimes.begin(),evolutionTimes.end());
	std::vector<double> volsDouble(vols.begin(),vols.end());

	Cube_gold<double> resD( FlatVolPseudoRootsExpCorr(rateTimesDouble,
		evolutionTimesDouble,
		beta,
		L,
		factors,
		volsDouble) );

	Cube_gold<float> res(CubeTypeConvert<float,double>(resD));

	return res;
}

Matrix_gold<double> GenerateExpLongCorrMatrix(const std::vector<double>& rateTimes,
											  double L,
											  double beta)
{
	int numberRates = rateTimes.size()-1;
	Matrix_gold<double> res(numberRates,numberRates,0.0);

	for (int i=0; i <numberRates; ++i)
	{
		res(i,i)=1.0;

		for (int j=0; j < i; ++j)
		{
			res(i,j) = res(j,i) = L + (1.0-L)*exp(-beta*fabs(rateTimes[i]-rateTimes[j]));

		}
	}
	return res;
}

Cube_gold<double> FlatVolPseudoRootsOfCovariances(const std::vector<double>& rateTimes,
												  const std::vector<double>& evolutionTimes,
												  const CubeConstFacade<double>& correlationMatrices,
												  const std::vector<double>& vols,
												  int factors,
												  int correlationStep
												  )
{

	if (evolutionTimes.size()*correlationStep != correlationMatrices.numberLayers() && correlationStep != 0 )
		GenerateError("evolutionTimes correlationPseudoRoots size mismatch in FlatVolPseudoRootsOfCovariances");

	if (rateTimes.size() != correlationMatrices.numberRows()+1)
		GenerateError("rateTimes correlationMatrices size mismatch in FlatVolPseudoRootsOfCovariances");

	if (vols.size() != correlationMatrices.numberRows())
		GenerateError("vols correlationMatrices size mismatch in FlatVolPseudoRootsOfCovariances");

	int numberRates = correlationMatrices.numberRows();
	if  (correlationMatrices.numberColumns() != numberRates)
		GenerateError("vols correlationMatrices now square in FlatVolPseudoRootsOfCovariances");


	int steps =  evolutionTimes.size();

	Cube_gold<double> pseudoRoots(steps,numberRates,factors,0.0);
	Matrix_gold<double> covariance(numberRates,numberRates,0.0);

	double startTime =0.0;

	for (int step=0; step < steps; ++step)
	{
		for (int r=0; r < numberRates; ++r)
		{
			double timeMovingR = std::min(evolutionTimes[step],rateTimes[r])-std::min(startTime,rateTimes[r]);

			covariance(r,r) = correlationMatrices(step*correlationStep,r,r) * vols[r]*vols[r]*timeMovingR; 


			for (int s=0; s <r; ++s)			
			{	
				double timeMovingS = std::min(evolutionTimes[step],rateTimes[s])-std::min(startTime,rateTimes[s]);
				double csr= correlationMatrices(step*correlationStep,r,s) * vols[r]*vols[s]*std::min(timeMovingR,timeMovingS); 

				covariance(r,s) =csr;
				covariance(s,r) =csr;		

			}
		}

		startTime = evolutionTimes[step];

		MatrixFacade<double> pseudoRootatStep(pseudoRoots[step]);
		pseudoSqrtSchur(covariance, pseudoRootatStep,  factors, true );
	}

	return pseudoRoots;

}

Cube_gold<double> FlatVolPseudoRootsOfCovariances(const std::vector<double>& rateTimes,
												  const std::vector<double>& evolutionTimes,
												  const std::vector<double>& vols,
												  int factors,
												  double L,
												  double beta
												  )
{
	Matrix_gold<double> correlationMatrix(GenerateExpLongCorrMatrix(rateTimes,L,beta));
	MatrixConstFacade<double> corrFacade(correlationMatrix.Facade());
	CubeConstFacade<double> correlation_cube(corrFacade);

	int correlationStep=0;

	return FlatVolPseudoRootsOfCovariances(rateTimes,
		evolutionTimes,
		correlation_cube,
		vols,
		factors,
		correlationStep);


}

namespace
{
	double tol = 1E-9;
	// fuzzy equals
	bool close(double x, double y)
	{
		return fabs(x-y) < tol;
	}
}

// PRIMITIVE
double Abcdprimitive(double a_, double b_, double c_, double d_, double t, double S, double T) 
{

	if (T<t || S<t) 
		return 0.0;

	if (close(c_,0.0)) 
	{
		double v = a_+d_;
		return t*(v*v+v*b_*S+v*b_*T-v*b_*t+b_*b_*S*T-0.5*b_*b_*t*(S+T)+b_*b_*t*t/3.0);
	}

	double k1=std::exp(c_*t), k2=std::exp(c_*S), k3=std::exp(c_*T);

	return (b_*b_*(-1 - 2*c_*c_*S*T - c_*(S + T)
		+ k1*k1*(1 + c_*(S + T - 2*t) + 2*c_*c_*(S - t)*(T - t)))
		+ 2*c_*c_*(2*d_*a_*(k2 + k3)*(k1 - 1)
		+a_*a_*(k1*k1 - 1)+2*c_*d_*d_*k2*k3*t)
		+ 2*b_*c_*(a_*(-1 - c_*(S + T) + k1*k1*(1 + c_*(S + T - 2*t)))
		-2*d_*(k3*(1 + c_*S) + k2*(1 + c_*T)
		- k1*k3*(1 + c_*(S - t))
		- k1*k2*(1 + c_*(T - t)))
		)
		) / (4*c_*c_*c_*k2*k3);
}

double covarianceABCD(double a_, double b_, double c_, double d_, double t1,double t2, double S, double T) 
{
	if (t1>t2)
		GenerateError("t1 and t2 wrong way round in covarianceABCD");

	double stopTime = std::min(std::min(S,T),t2);
	if (t1 >= stopTime)
		return 0.0;
	double p2= Abcdprimitive(a_,  b_,  c_, d_, t2,  S,  T) ;
	double p1= Abcdprimitive(a_,  b_,  c_, d_, t1,  S,  T) ;

	return p2-p1;
}

Cube_gold<double> ABCDLBetaPseudoRoots(double a, double b, double c, double d, 
								  const std::vector<double>& evolutionTimes,
								  const std::vector<double>& rateStarts,
								  const std::vector<double>& multipliers,
								  int factors,
								  double L,
							      double beta)
{
	

	int numberRates = rateStarts.size();
	int steps =  evolutionTimes.size();

	Cube_gold<double> pseudoRoots(steps,numberRates,factors,0.0);
	Matrix_gold<double> covariance(numberRates,numberRates,0.0);

	double startTime =0.0;

	for (int step=0; step < steps; ++step)
	{
			
		double stopTime = evolutionTimes[step];

		for (int r=0; r < numberRates; ++r)
		{
		
			double kr = multipliers[r];
			covariance(r,r) =covarianceABCD(a,b,c,d,startTime,stopTime,rateStarts[r],rateStarts[r])*kr*kr;

			for (int s=0; s <r; ++s)			
			{	
				double ks = multipliers[s];
				double rho = exp(-beta*fabs(rateStarts[s]-rateStarts[r]))*(1-L)+L;

				double csr = covarianceABCD(a,b,c,d,startTime,stopTime,rateStarts[r],rateStarts[s]);
				double covsr=csr*rho*ks*kr;

				covariance(r,s) =covsr;
				covariance(s,r) =covsr;		

			}
		}

		startTime = evolutionTimes[step];

		MatrixFacade<double> pseudoRootatStep(pseudoRoots[step]);
		pseudoSqrtSchur(covariance, pseudoRootatStep,  factors, true );
	}

	return pseudoRoots;
}




