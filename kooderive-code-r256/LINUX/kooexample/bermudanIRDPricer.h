//
//
//       Bermudan IRD Pricer.h
//
//
// (c) Mark Joshi 2011,2012,2013
// This code is released under the GNU public licence version 3

#ifndef BERMUDAN_IRD_PRICER_EXAMPLE_H
#define BERMUDAN_IRD_PRICER_EXAMPLE_H

void BermudanPricerExample(int pathsPerBatch,int numberOfBatches, int numberOfSecondPassBatches, bool useCula );

double BermudanMultiLSPricerExample(int pathsPerBatch,
								  int numberOfBatches, 
								  int numberOfSecondPassBatches, 
								  bool useCrossTerms,
								  int regressionDepth,
								  float sdsForCutOff,
								  int minPathsForRegression,
								  float lowerFrac,
								  float upperFrac,
								  float multiplier,
								  bool globallyNormalise,
								  int duplicate, // if zero use same paths as for first pass, if 1 start at the next paths
								  bool useLog,
								  int numberOfRates,
								  float firstForward,
								  float forwardIncrement,
								  float displacement,
								  float strike,
								  double beta,
								  double L,
								  double a, 
								  double b,
								  double c, 
								  double d,
								  int numberNonCallCoupons,
								  double firstRateTime,
								  double rateLength,
								  double payReceive_d,
								  bool useFlatVols,
								  bool annulNonCallCoupons,
								  double initialNumeraireValue,
								  bool globalDiscounting,
								  bool verbose,
								  int device,
								  int LMMthreads,
								  bool scrambleFirst,
								  bool scrambleSecond
								  );
#endif
