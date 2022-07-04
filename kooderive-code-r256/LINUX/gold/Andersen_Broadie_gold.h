//
//
//                          Andersen_Broadie_gold.h
//
//

#ifndef ANDERSEN_BROADIE_GOLD_H
#define ANDERSEN_BROADIE_GOLD_H

#include <gold/MatrixFacade.h>

void ABDualityGapEstimateZeroRebate(const MatrixConstFacade<int>& exerciseIndicators_mat,
        const MatrixConstFacade<double>& means_i_mat,
        const MatrixConstFacade<double>& sds_i_mat,
        double& mean,
        double& se,
        const std::vector<bool>& isExerciseDate_vec);

void ABDualityGapEstimateZeroRebateBiasEstimation(const MatrixConstFacade<int>& exerciseIndicators_mat,
                                                  const MatrixConstFacade<double>& means_i_mat,
                                                  const MatrixConstFacade<double>& sds_i_mat,
                                                  double& mean,
                                                  double& se,
                                                  double& mean2,
                                                  double& se2,
                                                  double& mean3,
                                                  double& se3,
                                                  double& mean4,
                                                  double& se4,
                                                  int pathsForGaussianEstimation,
                                                  int seed,
                                                  const std::vector<bool>& isExeriseDate_vec,
                                                  int numberExerciseDates);

#endif
