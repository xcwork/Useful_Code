//
//
//                          Andersen_Broadie_gold.h
//
//

#ifndef ANDERSEN_BROADIE_GOLD_H
#define ANDERSEN_BROADIE_GOLD_H

#include <gold/MatrixFacade.h>
#include <gold/math/basic_matrix_gold.h>

void ABDualityGapEstimateNonZeroRebate(const MatrixConstFacade<int>& exerciseIndicators_mat,
                                       const MatrixConstFacade<double>& deflated_payoffs_mat,
                                    const MatrixConstFacade<double>& means_i_mat,
                                    const MatrixConstFacade<double>& sds_i_mat,
                                    double& mean,
                                    double& se,
                                    const std::vector<bool>& isExeriseDate_vec,
                                    std::vector<double>& workspace_vec);

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
/*
void ABDualityGapEstimateGeneralRebateBiasEstimation(const MatrixConstFacade<int>& exerciseIndicators_mat,
                                                  const MatrixConstFacade<double>& means_i_mat,
                                                  const MatrixConstFacade<double>& sds_i_mat,
                                                  const MatrixConstFacade<double>& deflated_payoffs_mat,
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
                                                  int numberExerciseDates,
                                                  std::vector<double>& corrections2_vec,
                                                  std::vector<double>& corrections3_vec,
                                                  std::vector<double>& corrections4_vec
                                                  );
*/

void ABDualityGapEstimateGeneralRebateBiasEstimation(const MatrixConstFacade<int>& exerciseIndicators_mat,
                                                  const MatrixConstFacade<double>& means_i_mat,
                                                  const MatrixConstFacade<double>& sds_i_mat,
                                                  const MatrixConstFacade<double>& deflated_payoffs_mat,
                                                  double& mean,
                                                  double& se,
                                                  double& mean2,
                                                  double& se2,
                                                  double& mean3,
                                                  double& se3,
                                                  std::vector<double>& mean4,
                                                  std::vector<double>&  se4,
                                                  const std::vector<int>& pathsForGaussianEstimation_vec,
                                                  int seed,
                                                  const std::vector<bool>& isExeriseDate_vec,
                                                  int numberExerciseDates,
                                                  std::vector<double>& corrections2_vec,
                                                  std::vector<double>& corrections3_vec,
                                                  Matrix_gold<double>& corrections4_mat
                                                  );

void ComputeMultiplicativeHedgeVector(const MatrixConstFacade<int>& exerciseIndicators_mat,
                                       const MatrixConstFacade<double>& deflated_payoffs_mat,
                                       const MatrixConstFacade<double>& means_i_mat,
                                       MatrixFacade<double>& X_mat);


void ComputeABHedgeVector(const MatrixConstFacade<int>& exerciseIndicators_mat,
                                       const MatrixConstFacade<double>& deflated_payoffs_mat,
                                       const MatrixConstFacade<double>& means_i_mat,
                                       MatrixFacade<double>& M_mat);

void MixedDualityGapEstimateNonZeroRebate(
                                          const MatrixConstFacade<double>& deflated_payoffs_mat,
                                          const MatrixConstFacade<double>& M_mat,
                                          const MatrixConstFacade<double>& X_mat,
                                          double& mean,
                                          double& se,
                                          const std::vector<bool>& isExeriseDate_vec,
                                          std::vector<double>& pathwiseVals_vec);

void ComputeMixedHedgeVector(const MatrixConstFacade<int>& exerciseIndicators_mat,
                                       const MatrixConstFacade<double>& deflated_payoffs_mat,
                                       const MatrixConstFacade<double>& means_i_mat,
                                       double lowerBound,
                                       double theta,
                                       MatrixFacade<double>& X_mat);
#endif
