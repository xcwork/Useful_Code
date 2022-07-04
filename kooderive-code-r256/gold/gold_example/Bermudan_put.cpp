//
//											bermudan_put.cpp
//
//

// (c) Mark Joshi 2015
// released under GPL v 3.0
//#include "Bermudan_put.h"
#include <gold/gold_test/MultiDBS_gold_test.h>
#include <gold/MultiD_BS_evolver_classes_gold.h>
#include <gold/volstructs_gold.h>
#include <gold/MonteCarloStatistics_concrete_gold.h>
#include <gold/BSFormulas_gold.h>
#include <gold/EarlyExercisableMultiEquityPayoff_gold.h>
#include <gold/BasisVariableExtractionMultiEquity_gold.h>
#include <gold/volstructs_gold.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <gold/Regression_Selector_concrete_gold.h>
#include <gold/LS_basis_examples_gold.h>
#include <gold/ExerciseIndices_gold.h>
#include <gold/LMM_evolver_classes_gold.h>
#include <gold/LS_Basis_gold.h>
#include <gold/cashFlowGeneration_product_gold.h>
#include <gold/cashFlowDiscounting_gold.h>
#include <gold/CashFlowAggregation_gold.h>
#include <gold/MultiLS_regression.h>
#include <gold/cashFlowGeneration_earlyEx_product.h>
#include <gold/early_exercise_value_generator_gold.h>
#include <float.h>
#include <gold/Mersenne_gold.h>
#include <gold/LMM_evolver_partial_gold.h>
#include <gold/Andersen_Broadie_gold.h>
#include <gold\MultiD_Path_Gen_Partial_gold.h>



int BermudanPutCG(double r, double d, double sigma, double K, double S0, int N, int samples, double dt, int seed, double w1min, double w1max, double w2min, double w2max, int gridpoints1, int gridpoints2)
{

    int numberStocks =1;
    int factors=1;
    std::vector<double> initial_stocks_vec(1);
    initial_stocks_vec[0] =S0;
    double sd = sigma*sqrt(dt);    
    std::vector<double> sds_vec(N,sd);

    double drift = (r-d)*dt;
    std::vector<double> drifts_vec(N,drift);


    MultiDBS_PathGen_Partial_gold<double> generator( 
        numberStocks, 
        factors, 
        N, 
        sds_vec,
        drifts_vec
        );

    Cube_gold<double> variates(N,factors,samples);

    Cube_gold<double> paths(N,numberStocks,samples);

    MersenneTwisterUniformRng rng(seed);
    rng.populateCubeWithNormals<CubeFacade<double>,double>(variates.Facade());

    generator.getPaths(samples,
        0,
        0,
        initial_stocks_vec,
        variates.ConstFacade(),
        0, // specifies  how to associate normals in cube with steps in simulation
        paths.Facade());

    Matrix_gold<double> M_mat(samples,N,0.0);
    Matrix_gold<double> X_mat(samples,N,1.0);
    Matrix_gold<double> C_mat(samples,N,1.0);

    Matrix_gold<double> payoffs_mat(N,samples,1.0);

    double T = N*dt;
    int s=0;

    for (; s+1< N; ++s)
    {
        double t= (s+1.0)*dt;
        double tau = T-t;
        double df =exp(-r*t);

        for (int p=0; p < samples; ++p)
        {
            double St = paths(s,0,p);
            double BSPrice = BlackScholesPut(St,r,d,tau,sigma,K);
            M_mat(p,s) = BSPrice*df;
            X_mat(p,s) = BSPrice*df;
            double value = std::max(K-St,0.0);
            payoffs_mat(s,p)=value*df;

        }
    } 

    double df =exp(-r*T);
    for (int p=0; p < samples; ++p)
    {
        double St = paths(s,0,p);
        double value = std::max(K-St,0.0);
        M_mat(p,s) = value*df;
        X_mat(p,s) = value*df;
        payoffs_mat(s,p)=value*df;
    }

    std::vector<bool> exerciseable(N,true);

    double abMean, abse;
    std::vector<double> pathwiseAB_Vec(samples);
    MixedDualityGapEstimateNonZeroRebate(
        payoffs_mat,
        M_mat,
        C_mat,
        abMean,
        abse,
        exerciseable,
        pathwiseAB_Vec);

    double mixedMean, mixedse;
    std::vector<double> pathwiseMixed_Vec(samples);
    MixedDualityGapEstimateNonZeroRebate(
        payoffs_mat,
        M_mat,
        X_mat,
        mixedMean,
        mixedse,
        exerciseable,
        pathwiseMixed_Vec);

    double mmMean, mmse;
    std::vector<double> pathwiseMM_Vec(samples);
    MixedDualityGapEstimateNonZeroRebate(
        payoffs_mat,
        X_mat,
        X_mat,
        mmMean,
        mmse,
        exerciseable,
        pathwiseMM_Vec);
    double mcMean, mcse;
    std::vector<double> pathwiseMC_Vec(samples);
    MixedDualityGapEstimateNonZeroRebate(
        payoffs_mat,
        X_mat,
        C_mat,
        mcMean,
        mcse,
        exerciseable,
        pathwiseMC_Vec);

    std::cout << " AB mean and se," << abMean << "," << abse << "," << " mixed mean and se, " 
        << mixedMean << "," << mixedse << ", mult mean and se , " <<  mmMean << "," << mmse << ", mult plus const mean and se , "<< mcMean <<","<< mcse << "\n";

    double BSPrice = BlackScholesPut(S0,r,d,T,sigma,K);

    std::cout << " black scholes price " << BSPrice << "\n";
    std::vector<double> pathwiseMCW_Vec(samples);

    Matrix_gold<double> MW_mat(samples,N,0.0);
    Matrix_gold<double> XW_mat(samples,N,1.0);

    double best =10000000;
    double w1b, w2b, wbse;
    for (int i=0; i <= gridpoints1; ++i)
        for (int j=0; j <= gridpoints2; ++j)
        {
            double w1 = w1min+(w1max-w1min)*(i+0.0)/(gridpoints1);
            double w2 = w2min+(w2max-w2min)*(j+0.0)/(gridpoints2);

            for (int s=0; s< N; ++s)
                for (int p=0; p < samples; ++p)
                {
                    MW_mat(p,s) = w1*M_mat(p,s);
                    XW_mat(p,s) = w2+(1-w2)*X_mat(p,s);

                }

                double wMean, wse;
                MixedDualityGapEstimateNonZeroRebate(
                    payoffs_mat,
                    MW_mat,
                    XW_mat,
                    wMean,
                    wse,
                    exerciseable,
                    pathwiseMCW_Vec);
                std::cout << w1 << "," << w2 << "," << wMean+w1*BSPrice << "," << wse << "\n";

                double val = wMean+w1*BSPrice;

                if ( val < best)
                {
                    best = val;
                    w1b = w1;
                    w2b= w2;
                    wbse = wse;
                }
        }

        std::cout << "best, " <<w1b << "," << w2b << "," << best << "," << wbse << "\n";

         std::cout << " AB mean and se," << abMean+BSPrice <<  ", mult mean and se , " <<  mmMean+BSPrice << "," << mmse << "\n";
        return 0;

}


int BermudanPut()
{
    for (int j=50; j <= 100; j=j+5)
    {
        double S0 = j+0.0;
        std::cout << "\nSpot," << S0 << "\n";
        BermudanPutCG(0.2, 0.0, 0.3, 100,S0, 100, 1000000, 0.01, 4643543,1,1,0.0,1.0,1,100);
    }
    return 0;
}