
//
//
//                  MultiLS_regression.h
//
//
// (c) Mark Joshi  2013
// This code is released under the GNU public licence version 3

#ifndef MULTILS_REGRESSION_H
#define MULTILS_REGRESSION_H
#include <algorithm>
#include <vector>
#include <gold/MatrixFacade.h>
#include <gold/math/typedefs_math_gold.h>
#include <gold/LS_regression.h>
#include <gold/Regression_Selector_gold.h>

double generateRegressionCoefficientsViaLSMultiQuadratic_gold(int numberExerciseDates,
												std::vector<double>& products_cube_vec,
												std::vector<double>& targets_mat_vec,
												CubeFacade<double>& regression_coefficients_cube,
												MatrixFacade<double>& lowerCuts_mat,
												MatrixFacade<double>& upperCuts_mat,
												const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
												const std::vector<int>& basisVariablesEachStep,
												int maxBasisVariables,
												const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
												const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, // deflated to current exercise time
												const MatrixFacade<Realv>& numeraireValues_matrix,
												const std::vector<int>& deflation_locations_vec,
												int paths,
												int maxNumberOfExtraRegressions,
												int lowerPathCutoff,
												RegressionSelectorGold& regressionSelector,
												std::vector<Realv>& nextStepWorkspace_vec,
												std::vector<Realv>& basisVariableWorkspace_vec,
												std::vector<Realv>& basisFunctionsWorkspace_vec,
												std::vector<double>& regressionMatrixWorkspace_vec,
												std::vector<double>& sumProductsWorkspace_vec,
												std::vector<double>& meansWorkspace_vec,
												std::vector<double>& sdsWorkspace_vec, 
												bool useCrossTerms);


template<class T>
void updateDeflatedNextStepValuesMulti_gold(const CubeFacade<double>& regressionWeights,
											const MatrixFacade<double>& lowerCuts_mat,
											const MatrixFacade<double>& upperCuts_mat,
											const CubeFacade<Realv>& means_cube,
											const CubeFacade<Realv>& sds_cube,
											int maxNumberOfRegressions,
									   Realv AndersenShift,
									   const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
									   const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
									   std::vector<Realv>& deflatedNextStepValues_vec, // deflated to current exercise time
									   const MatrixConstFacade<Realv>& deflatedExerciseValue_matrix,
									   int stepNumber,
									   int totalPaths,
									   int numberVariablesThisStep,
									   T& generator,
									   std::vector<Realv>& workSpace1_vec,
									   std::vector<Realv>& workSpace2_vec
									   )
{
	workSpace1_vec.resize(numberVariablesThisStep);
	int numberDataPts= generator.numberDataPoints();
	workSpace2_vec.resize(numberDataPts);

//	std::cout << "\n\nupdateDeflatedNextStepValuesMulti_gold\n step" << stepNumber << "\n";
	for (int p=0; p < totalPaths; ++p)
	{
	
		Realv estContinuation = 0.0;
		Realv netContinuation = 0.0;
		Realv exValue = deflatedExerciseValue_matrix(stepNumber,p);

		int depth =0;

		while (depth == 0 
			  || (depth < maxNumberOfRegressions && netContinuation > lowerCuts_mat(stepNumber,depth-1) && netContinuation < upperCuts_mat(stepNumber,depth-1) ))
		{
			for (int i=0; i < numberVariablesThisStep; ++i)
			{
				double x= basisFunctionVariables_cube(stepNumber,i,p);
				double m = means_cube(stepNumber,depth,i);
				double s= sds_cube(stepNumber,depth,i);
				workSpace1_vec[i] =(x-m)/s;
			}
			generator.writeData(workSpace1_vec, workSpace2_vec);

			estContinuation = AndersenShift;

			for (int j=0; j < numberDataPts; ++j)
				estContinuation+= static_cast<Realv>(workSpace2_vec[j]*regressionWeights(stepNumber,depth,j));
			
			netContinuation =estContinuation - exValue;
			
			++depth;
		}

	//	std::cout << p << " , " << estContinuation << " , " << depth << " , ";

		if (exValue > estContinuation)
		{ // exercise
			deflatedNextStepValues_vec[p] = exValue;
	//		std::cout << 1 << "\n";
		}
		else
		{ // don't exercise 
			deflatedNextStepValues_vec[p] += deflatedCashFlowValues_matrix(stepNumber,p);
//			std::cout << -1 << "\n";
		}

	}

}

template<class T>
class MultiLSExerciseStrategy
{
public:
	MultiLSExerciseStrategy(const CubeConstFacade<Realv>& LS_coefficients_cube,
	                        const MatrixConstFacade<Realv>& lowerCuts,
                            const MatrixConstFacade<Realv>& upperCuts,
							const CubeConstFacade<Realv>& means_variables_cube,
							const CubeConstFacade<Realv>& sds_variables_cube,	                        
			                int maxRegressionDepth,
						    const std::vector<Realv>& andersenShifts_vec,
							const std::vector<int>& numberVariables_vec,
							int steps);

	bool decide(double exValue, const CubeConstFacade<Realv>& dataVariables_cube,int e,int p);
												  
private:

	int steps_;
	int maxRegressionDepth_;
	int maxVariableSize_;
	int maxDataSize_;

	std::vector<Realv> andersenShifts_vec_;
    std::vector<int> numberVariables_vec_;

	// this data is not copied so make sure that it stays in scope 
	CubeConstFacade<Realv> LS_coefficients_cube_;
	MatrixConstFacade<Realv> lowerCuts_;
    MatrixConstFacade<Realv> upperCuts_;
	CubeConstFacade<Realv> means_variables_cube_;
	CubeConstFacade<Realv> sds_variables_cube_;

	std::vector<T> variableExpanders_;
	std::vector<Realv> wspace1_vec_;
	std::vector<Realv> wspace2_vec_;
};

template<class T>
MultiLSExerciseStrategy<T>::MultiLSExerciseStrategy(const CubeConstFacade<Realv>& LS_coefficients_cube,
	                        const MatrixConstFacade<Realv>& lowerCuts,
                            const MatrixConstFacade<Realv>& upperCuts,
							const CubeConstFacade<Realv>& means_variables_cube,
							const CubeConstFacade<Realv>& sds_variables_cube,
			                int maxRegressionDepth,
						    const std::vector<Realv>& andersenShifts_vec,
							const std::vector<int>& numberVariables_vec,
							int steps)
							: 
                            steps_(steps),
	maxRegressionDepth_(maxRegressionDepth),
	LS_coefficients_cube_(LS_coefficients_cube),
	lowerCuts_(lowerCuts),
    upperCuts_(upperCuts),
	means_variables_cube_(means_variables_cube),
	sds_variables_cube_(sds_variables_cube),
	andersenShifts_vec_(andersenShifts_vec),
	numberVariables_vec_(numberVariables_vec),
	variableExpanders_(0)
{
	maxVariableSize_ = *std::max_element(numberVariables_vec.begin(),numberVariables_vec.end());
	T obj(maxVariableSize_);

	maxDataSize_ = obj.numberDataPoints();

	variableExpanders_.reserve(steps);

	for (int i=0; i < steps; ++i)
		variableExpanders_.push_back(T(numberVariables_vec_[i]));

	wspace1_vec_.resize(maxVariableSize_);
	wspace2_vec_.resize(maxDataSize_);


}
template<class T>
bool MultiLSExerciseStrategy<T>::decide(double exValue, 
										const CubeConstFacade<Realv>& dataVariables_cube,
										int e,
										int p)
{
	

	int depth=0;
	double netValue;

	double lowerCut, upperCut;

	do
	{
		for (int i=0; i < numberVariables_vec_[e]; ++i)
		    wspace1_vec_[i] = (dataVariables_cube(e,i,p)-means_variables_cube_(e,depth,i))/sds_variables_cube_(e,depth,i);

		variableExpanders_[e].writeData(wspace1_vec_, wspace2_vec_);

		double continuationValue = andersenShifts_vec_[e];

		for (int i=0; i < variableExpanders_[e].numberDataPoints(); ++i)
		{
			double var = wspace2_vec_[i];
			double coeff = LS_coefficients_cube_(e,depth,i);
			continuationValue += var*coeff;
		}
		netValue = continuationValue- exValue;

		lowerCut = lowerCuts_(e,depth);
		upperCut = upperCuts_(e,depth);

		++depth;

	}
	while (depth < maxRegressionDepth_ && netValue > lowerCut && netValue < upperCut);


	return netValue <0.0;
}


template<class T>
double SecondPassPriceLSAMultiUsingAggregatedFlows_gold(int numberExerciseDates,
												    const CubeConstFacade<Realv>& dataVariables_cube,
												   int maxBasisVariables,
												   const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, 
												   const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, 
												   const MatrixFacade<Realv>& numeraireValues_matrix,
												   int paths,
								//				   const std::vector<int>& deflationLocations_vec,
												   T& exerciseStrategy
												   )
{

	

	Realv result =0.0;

	for (int p=0; p < paths; ++p)
	{
		Realv totalNumerairesThisPath =0.0;
		bool pathNotDone = true;
		for (int e=0; e < numberExerciseDates && pathNotDone ; ++e)
		{
			
			double exValue = deflatedExerciseValues_matrix(e,p);
			double cfValue = deflatedCashFlowValues_matrix(e,p);


			bool exerciseNow = exerciseStrategy.decide(exValue,dataVariables_cube,e,p);


			if (!exerciseNow)
			{
				totalNumerairesThisPath += cfValue/numeraireValues_matrix(e,p);
			}
			else
			{
				totalNumerairesThisPath += exValue/numeraireValues_matrix(e,p);
				pathNotDone = false;
			}


		}

		result += totalNumerairesThisPath;


	}

	result/=paths;

	return result;
}


double generateRegressionCoefficientsViaLSMultiQuadratic_gold(int numberExerciseDates,
															  std::vector<double>& products_cube_vec,
															  std::vector<double>& targets_mat_vec,
															  CubeFacade<double>& regression_coefficients_cube,
															  MatrixFacade<double>& lowerCuts_mat,
															  MatrixFacade<double>& upperCuts_mat,
															  CubeFacade<Realv>& means_variables_cube,
															  CubeFacade<Realv>& sds_variables_cube,
															  bool normalise,
															  const CubeConstFacade<Realv>&  basisFunctionVariables_cube,
															  const std::vector<int>& basisVariablesEachStep,
															  int maxBasisVariables,
															  const MatrixConstFacade<Realv>& deflatedCashFlowValues_matrix, // deflated to current exercise time
															  const MatrixConstFacade<Realv>& deflatedExerciseValues_matrix, // deflated to current exercise time
															  const MatrixFacade<Realv>& numeraireValues_matrix,
												//			  const std::vector<int>& deflation_locations_vec,
															  int paths,
															  int maxNumberOfExtraRegressions,
															  int lowerPathCutoff,
															  RegressionSelectorGold& regressionSelector,
															  bool useCrossTerms);



#endif
