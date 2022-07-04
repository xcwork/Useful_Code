//
//											volstructs_gold.h
//
//

// (c) Mark Joshi 2013
// released under GP L v 3.0

#ifndef VOLSTRUCTS_GOLD_H
#define VOLSTRUCTS_GOLD_H

#include <gold/math/cube_gold.h>
#include <gold/math/basic_matrix_gold.h>
#include <vector>

// given the pseudo roots of the correlation matrices
// produce the pseudo roots of the covariance matrices
//for a flat vol structure
Cube_gold<double> FlatVolPseudoRoots(const std::vector<double>& rateTimes,
									 const std::vector<double>& evolutionTimes,
									 const CubeConstFacade<double>& correlationPseudoRoots,
									 const std::vector<double>& vols);

// produce the pseudo roots of the correlation matrices
// for L+(1-L)exp(-beta|t_i-t_j|) form
// reduced factor model
// each step gives the same matrix
Cube_gold<double> ExponentialLongCorrelationPseudoRoots(const std::vector<double>& rateTimes, 
														int numberSteps,
														double beta,
														double L,
														int factors);
// join of two previous functions for time saving
Cube_gold<double> FlatVolPseudoRootsExpCorr(const std::vector<double>& rateTimes,
									 const std::vector<double>& evolutionTimes,
									 double beta,
									 double L,
									 int factors,
									 const std::vector<double>& vols);

Cube_gold<float> FlatVolPseudoRootsFloat(const std::vector<float>& rateTimes,
									 const std::vector<float>& evolutionTimes,
									 float beta,
									 float L,
									 int factors,
									 const std::vector<float>& vols);

Matrix_gold<double> GenerateExpLongCorrMatrix(const std::vector<double>& rateTimes,
											  double L,
											  double beta);

Cube_gold<double> FlatVolPseudoRootsOfCovariances(const std::vector<double>& rateTimes,
								const std::vector<double>& evolutionTimes,
								const CubeConstFacade<double>& correlationMatrices,
								const std::vector<double>& vols,
								int factors,
								int correlationStep
								);

Cube_gold<double> FlatVolPseudoRootsOfCovariances(const std::vector<double>& rateTimes,
								const std::vector<double>& evolutionTimes,
								const std::vector<double>& vols,
								int factors,
								double L,
								double beta
								);
double covarianceABCD(double a_, double b_, double c_, double d_, double t1,double t2, double S, double T);
double Abcdprimitive(double a_, double b_, double c_, double d_, double t, double S, double T);

Cube_gold<double> ABCDLBetaPseudoRoots(double a, double b, double c, double d, 
								  const std::vector<double>& evolutionTimes,
								  const std::vector<double>& rateStarts,
								  const std::vector<double>& multipliers,
								  int factors,
								  double L,
							      double beta);

#endif

