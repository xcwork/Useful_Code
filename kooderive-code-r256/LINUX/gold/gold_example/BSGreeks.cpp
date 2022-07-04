//
//
//                  BSGreeks.cpp
//
//
// copyright Mark Joshi 2012
// released under GPL V3.0

#include <gold/math/typedefs_math_gold.h>
#include <gold/Bridge_gold.h>
#include <gold/sobol_gold.h>
#include <gold/MatrixFacade.h>
#include <gold/math/Normals_gold.h>
#include <gold/BSFormulas_gold.h>
#include <gold/mixedLRPathwise_gold.h>
#include <gold/likelihoodRatio_gold.h> 
#include <gold/ComputeStatistics_gold.h>
#include <gold/pathwise_gold.h>
#include <gold/oppGreeksBS_gold.h> 
#include <gold/Mersenne_gold.h>
#include <gold/scramble_gold.h>
#include <gold/MonteCarloStatistics_concrete_gold.h>
#include <ctime>
#include <algorithm>
#include <iostream>

void GetBSParameters(int dataSet, Realv& S0,  Realv& sigma, Realv& r, Realv& d, Realv& T, int& log2steps)
{
    //  if (dataSet ==0)
    //  {
    S0 =80+5*dataSet;
    sigma =0.2;
    r = 0.0;
    d = 0.0;
    T = 0.25;
    log2steps =0;
    // }

    /*  if (dataSet ==1)
    {
    S0 =100;
    sigma =0.2;
    r = 0.05;
    d = 0.02;
    T = 0.01;
    log2steps = 2;
    }
    */
}

void GetSobolUniforms(std::vector<Realv>& uniforms_vec,                  
                      int &normalOffset,
                      int &n_vectors,
                      int paths, 
                      int n_steps)

{
    normalOffset =1;
    n_vectors =  paths+normalOffset;
    int n_dimensions = n_steps;

    std::vector<unsigned int> directions_vec(n_dimensions*n_directions);
    initSobolDirectionVectors(n_dimensions, &directions_vec[0]);
    uniforms_vec.resize(n_vectors*n_steps);

    sobolCPU(n_vectors,  n_dimensions,  &directions_vec[0], &uniforms_vec[0]);
}

void GetSobolUniformsWithScrambling(std::vector<Realv>& uniforms_vec,                  
                                    int &normalOffset,
                                    int &n_vectors,
                                    int paths, 
                                    int n_steps,
                                    const std::vector<unsigned int>& scramblingVector)

{
    normalOffset =1;
    n_vectors =  paths+normalOffset;
    int n = n_vectors;
    int n_dimensions = n_steps;

    std::vector<unsigned int> directions_vec(n_dimensions*n_directions);
    initSobolDirectionVectors(n_dimensions, &directions_vec[0]);
    uniforms_vec.resize(n_vectors*n_steps);

    std::vector<unsigned int> sobolint_vec(uniforms_vec.size());
    std::vector<unsigned int> scrambled_vec(uniforms_vec.size());

    sobolCPUintsOffset( n, n_dimensions, 0, &directions_vec[0], &sobolint_vec[0]);

    scramble_gold( sobolint_vec, // data to scramble
        scrambled_vec, //  scrambled data
        scramblingVector,
        n_dimensions,
        paths);

    for (size_t t=0; t < uniforms_vec.size(); ++t)
        uniforms_vec[t] =(scrambled_vec[t]    + 0.5)/4294967296.0;



}

void GetSobolPaths(std::vector<Realv>& uniforms_vec,
                   std::vector<Realv>& normals_vec,
                   int &normalOffset,
                   int &n_vectors,
                   int paths, 
                   int n_steps)

{
    GetSobolUniforms( uniforms_vec,                  
        normalOffset,
        n_vectors,
        paths, 
        n_steps);
    normals_vec.resize(n_vectors*n_steps);

    inverseCumulativeGold<Realv> inv;
    std::transform(uniforms_vec.begin(), uniforms_vec.end(),normals_vec.begin(),inv);

}

void GenerateBSPathDataGivenVariates(Realv S0,  Realv sigma, Realv r, Realv d, Realv T,  
                                     int paths,
                                     int n_steps, 
                                     int normaloffset, 
                                     Realv& dt, 
                                     Realv& drift_per_step, 
                                     Realv& stdev_per_step,
                                     int n_vectors, 
                                     const std::vector<Realv>& uniforms_vec,
                                     const std::vector<Realv>& normals_vec,
                                     std::vector<Realv>& logSpots_vec,
                                     std::vector<Realv>& spots_vec,
                                     std::vector<Realv>& drifts_vec,
                                     std::vector<Realv>& stdev_vec,
                                     std::vector<Realv>& dTs_vec,
                                     std::vector<Realv>& sigmas_vec)
{



    dt = T/n_steps;

    drift_per_step = dt*(r-d-0.5*sigma*sigma);
    stdev_per_step = sigma*sqrt(dt);

    drifts_vec.resize(n_steps,drift_per_step);
    stdev_vec.resize(n_steps,stdev_per_step);
    dTs_vec.resize(n_steps,dt);
    sigmas_vec.resize(n_steps,sigma);


    logSpots_vec.resize(paths*n_steps);
    spots_vec.resize(paths*n_steps);

    MatrixConstFacade<Realv> uniforms_mat(&uniforms_vec[0],n_steps,n_vectors);
    MatrixConstFacade<Realv> normals_mat(&normals_vec[0],n_steps,n_vectors);
    MatrixFacade<Realv> logSpots_mat(&logSpots_vec[0],paths,n_steps);
    MatrixFacade<Realv> spots_mat(&spots_vec[0],paths,n_steps);


    for (int p=0; p < paths; ++p)
    {

        Realv logst = log(S0);

        for (int i=0; i < n_steps; ++i)
        {
            Realv gaussian = normals_mat(i,p+1);
            logst += drifts_vec[i]+gaussian*stdev_vec[i];

            logSpots_mat(p,i) = logst;
            spots_mat(p,i) = exp(logst);
        }                          
    }


}


void GenerateBSPathData(Realv S0,  Realv sigma, Realv r, Realv d, Realv T,  
                        int paths,
                        int n_steps, 
                        int& normaloffset, 
                        Realv& dt, 
                        Realv& drift_per_step, 
                        Realv& stdev_per_step,
                        int& n_vectors, 
                        std::vector<Realv>& uniforms_vec,
                        std::vector<Realv>& normals_vec,
                        std::vector<Realv>& logSpots_vec,
                        std::vector<Realv>& spots_vec,
                        std::vector<Realv>& drifts_vec,
                        std::vector<Realv>& stdev_vec,
                        std::vector<Realv>& dTs_vec,
                        std::vector<Realv>& sigmas_vec)
{



    GetSobolPaths(uniforms_vec,
        normals_vec,
        normaloffset,
        n_vectors,
        paths, 
        n_steps);

    GenerateBSPathDataGivenVariates( S0,   sigma,  r,  d,  T,  
        paths,
        n_steps, 
        normaloffset, 
        dt, 
        drift_per_step, 
        stdev_per_step,
        n_vectors, 
        uniforms_vec,
        normals_vec,
        logSpots_vec,
        spots_vec,
        drifts_vec,
        stdev_vec,
        dTs_vec,
        sigmas_vec);
}

void GenerateCashFlowsPowerCalls(int numberFlows,  std::vector<Realv>& discountedCashFlows,
                                 std::vector<Realv>& discountedCashFlowStateDerivatives ,
                                 std::vector<int>& cashFlowGenerationIndices_vec,
                                 int power,
                                 int paths, 
                                 Realv K,
                                 Realv T,
                                 Realv r,
                                 int n_steps,
                                 const MatrixConstFacade<Realv>& spots_mat
                                 )
{

    discountedCashFlows.resize(paths*numberFlows);
    std::fill(discountedCashFlows.begin(),discountedCashFlows.end(),0.0);
    discountedCashFlowStateDerivatives.resize(paths*numberFlows);
    std::fill(discountedCashFlowStateDerivatives.begin(),discountedCashFlowStateDerivatives.end(),0.0);


    MatrixFacade<Realv> discountedCashFlows_mat(&discountedCashFlows[0],paths,numberFlows);
    MatrixFacade<Realv> discountedCashFlowStateDerivatives_mat(&discountedCashFlowStateDerivatives[0],paths,numberFlows);

    cashFlowGenerationIndices_vec.resize(paths*numberFlows);
    MatrixFacade<int> cashFlowGenerationIndices_mat(&cashFlowGenerationIndices_vec[0],paths,numberFlows);

    for (int p=0; p < paths; ++p)
    {
        for (int f=0; f < numberFlows; ++f)
        {
            int i = (f+1)*n_steps/numberFlows-1;
            cashFlowGenerationIndices_mat(p,f) = i;
            Realv callPayoff= spots_mat(p,i) - K;

            if (callPayoff >0)
                discountedCashFlows_mat(p,f) = pow(callPayoff,power)*exp(-r*T*(f+1.0)/numberFlows);

            if (callPayoff >0.0 && power > 0)
                discountedCashFlowStateDerivatives_mat(p,f) = power*pow(callPayoff,power-1)*exp(-r*T);
        }
    }

}

void GenerateCashFlowsDoubleDigitals(int numberFlows,  std::vector<Realv>& discountedCashFlows,
                                     std::vector<Realv>& discountedCashFlowStateDerivatives ,
                                     std::vector<int>& cashFlowGenerationIndices_vec,
                                     int paths, 
                                     Realv K1,
                                     Realv K2,
                                     Realv T,
                                     Realv r,
                                     int n_steps,
                                     const MatrixConstFacade<Realv>& spots_mat
                                     )
{

    discountedCashFlows.resize(paths*numberFlows);
    std::fill(discountedCashFlows.begin(),discountedCashFlows.end(),0.0);
    discountedCashFlowStateDerivatives.resize(paths*numberFlows);
    std::fill(discountedCashFlowStateDerivatives.begin(),discountedCashFlowStateDerivatives.end(),0.0);


    MatrixFacade<Realv> discountedCashFlows_mat(&discountedCashFlows[0],paths,numberFlows);
    MatrixFacade<Realv> discountedCashFlowStateDerivatives_mat(&discountedCashFlowStateDerivatives[0],paths,numberFlows);

    cashFlowGenerationIndices_vec.resize(paths*numberFlows);
    MatrixFacade<int> cashFlowGenerationIndices_mat(&cashFlowGenerationIndices_vec[0],paths,numberFlows);

    for (int p=0; p < paths; ++p)
    {
        for (int f=0; f < numberFlows; ++f)
        {
            int i = (f+1)*n_steps/numberFlows-1;
            cashFlowGenerationIndices_mat(p,f) = i;
            Realv s= spots_mat(p,i);

            if (K1 < s && s < K2)
                discountedCashFlows_mat(p,f) = exp(-r*T*(f+1.0)/numberFlows);

        }
    }

}


void GenerateCashFlowsTriggerableRangeAccrual( 
    std::vector<Realv>& discountedCashFlows,
    std::vector<Realv>& discountedCashFlowStateDerivatives ,
    std::vector<int>& cashFlowGenerationIndices_vec,
    int paths, 
    Realv accrualBarrier,
    Realv triggerLevel,
    Realv callNotional,
    Realv callStrike,
    int daysForTriggering,
    Realv rebate,
    Realv couponMaxRate,
    int daysPerPeriod,
    int numberPeriods,
    Realv T,
    Realv r,                                         
    const MatrixConstFacade<Realv>& spots_mat
    )
{
    int n_steps = daysPerPeriod*numberPeriods;
    Realv couponPerDay = couponMaxRate*T/n_steps;

    discountedCashFlows.resize(paths*n_steps);
    std::fill(discountedCashFlows.begin(),discountedCashFlows.end(),0.0);
    discountedCashFlowStateDerivatives.resize(paths*n_steps);
    std::fill(discountedCashFlowStateDerivatives.begin(),discountedCashFlowStateDerivatives.end(),0.0);


    MatrixFacade<Realv> discountedCashFlows_mat(&discountedCashFlows[0],paths,n_steps);
    MatrixFacade<Realv> discountedCashFlowStateDerivatives_mat(&discountedCashFlowStateDerivatives[0],paths,n_steps);

    cashFlowGenerationIndices_vec.resize(paths*n_steps);
    MatrixFacade<int> cashFlowGenerationIndices_mat(&cashFlowGenerationIndices_vec[0],paths,n_steps);

    for (int p=0; p < paths; ++p)
    {
        bool triggered=false;
        int step=0;
        for (int period =0; period < numberPeriods && !triggered; ++period)
        {
            int daysBehindBarrier =0;
            int day=0;
            Realv s_t;

            for (; day < daysPerPeriod; ++day, ++step)
            {
                cashFlowGenerationIndices_mat(p,step) = step;
                s_t = spots_mat(p,step);
                discountedCashFlows_mat(p,step) =  s_t < accrualBarrier ? couponPerDay*exp(-(r*T*(step+1.0))/ n_steps) : 0;

                if (s_t > triggerLevel)
                {
                    ++daysBehindBarrier;
                }
            }

            if (daysBehindBarrier >= daysForTriggering)
            {
                triggered=true;
                discountedCashFlows_mat(p,step-1) += (rebate+callNotional*(s_t - callStrike)) * exp((-r*T*step)/ n_steps);

            }

        }
    }

}


void GenerateCashFlowsCorridor( 
                               std::vector<Realv>& discountedCashFlows,
                               std::vector<Realv>& discountedCashFlowStateDerivatives ,
                               std::vector<int>& cashFlowGenerationIndices_vec,
                               int paths, 
                               Realv Lower_Barrier,
                               Realv Upper_Barrier,
                               int daysForTriggering,
                               Realv rebate,
                               int days,
                               Realv T,
                               Realv r,                                         
                               const MatrixConstFacade<Realv>& spots_mat,
                               bool consecutive
                               )
{
    int n_steps = days;

    discountedCashFlows.resize(paths);
    std::fill(discountedCashFlows.begin(),discountedCashFlows.end(),0.0);
    discountedCashFlowStateDerivatives.resize(paths);
    std::fill(discountedCashFlowStateDerivatives.begin(),discountedCashFlowStateDerivatives.end(),0.0);



    cashFlowGenerationIndices_vec.resize(paths);
    std::fill(cashFlowGenerationIndices_vec.begin(),cashFlowGenerationIndices_vec.end(),days-1);

   

    for (int p=0; p < paths; ++p)
    {
        bool breached=false;

        int daysBehindBarrier =0;

        for (int day=0; day<days && !breached; ++ day)
        {
            Realv s_t;
            s_t = spots_mat(p,day);

            if (s_t < Lower_Barrier || s_t > Upper_Barrier)
            {
                ++daysBehindBarrier;
                if (daysBehindBarrier>= daysForTriggering)
                {
                    breached = true;

                }
            }
            else 
                if (consecutive)
                    daysBehindBarrier =0;

        }


        if (!breached)
        {
            discountedCashFlows[p] = rebate*exp(-r*T);
        }
    }
}





int OneDBSMonteCarloPrice(int paths, int verbose, int dataset, Realv K)
{

    Realv S0 ;
    Realv sigma;
    Realv r;
    Realv d ;
    Realv T;
    int n_poweroftwo;
    GetBSParameters(dataset,  S0,    sigma,  r,  d, T,n_poweroftwo);


    int n_steps = intPower(2,n_poweroftwo);

    Realv dt = T/n_steps;

    Realv drift_per_step = dt*(r-d-0.5*sigma*sigma);
    Realv stdev_per_step = sigma*sqrt(dt);

    std::vector<Realv> drifts_vec(n_steps,drift_per_step);
    std::vector<Realv> stdev_vec(n_steps,stdev_per_step);

    int n_vectors =  paths+1;
    int n_dimensions = n_steps;

    std::vector<unsigned int> directions_vec(n_dimensions*n_directions);
    initSobolDirectionVectors(n_dimensions, &directions_vec[0]);

    std::vector<Realv> uniforms_vec(n_vectors*n_steps);
    std::vector<Realv> normals_vec(n_vectors*n_steps);
    std::vector<Realv> logSpots_vec(paths*n_steps);
    std::vector<Realv> spots_vec(paths*n_steps);

    MatrixConstFacade<Realv> uniforms_mat(&uniforms_vec[0],n_steps,n_vectors);
    MatrixConstFacade<Realv> normals_mat(&normals_vec[0],n_steps,n_vectors);
    MatrixFacade<Realv> logSpots_mat(&logSpots_vec[0],paths,n_steps);
    MatrixFacade<Realv> spots_mat(&spots_vec[0],paths,n_steps);

    sobolCPU(n_vectors,  n_dimensions,  &directions_vec[0], &uniforms_vec[0]);

    inverseCumulativeGold<Realv> inv;

    std::transform(uniforms_vec.begin(), uniforms_vec.end(),normals_vec.begin(),inv);

    for (int p=0; p < paths; ++p)
    {

        Realv logst = log(S0);

        for (int i=0; i < n_steps; ++i)
        {
            Realv gaussian = normals_mat(i,p+1);
            logst += drifts_vec[i]+gaussian*stdev_vec[i];

            logSpots_mat(p,i) = logst;
            spots_mat(p,i) = exp(logst);
        }                          
    }

    {
        Realv sum=0.0;
        Realv sumsq=0.0;

        for (int p=0; p < paths; ++p)
        {
            Realv callPayoff= std::max<Realv>(spots_mat(p,n_steps-1) - K,0)*exp(-r*T);
            sum += callPayoff;
            sumsq += callPayoff*callPayoff;

        }

        Realv callOptionEstimate = sum/paths;
        Realv callOptionVariance = sumsq/paths-callOptionEstimate*callOptionEstimate;
        Realv sterr = sqrt(callOptionVariance/paths);

        Realv BSCallPrice = BlackScholesCall(S0, r,  d,  T, sigma,  K);

        std::cout << " Call option estimated price " << callOptionEstimate << " with standard error " << sterr << "\n";
        std::cout << " Call option formula price " << BSCallPrice << "\n";
    }

    {
        Realv sum=0.0;
        Realv sumsq=0.0;

        for (int p=0; p < paths; ++p)
        {
            Realv putPayoff= std::max<Realv>(K-spots_mat(p,n_steps-1),0)*exp(-r*T);
            sum += putPayoff;
            sumsq += putPayoff*putPayoff;

        }

        Realv putOptionEstimate = sum/paths;
        Realv putOptionVariance = sumsq/paths-putOptionEstimate*putOptionEstimate;
        Realv sterr = sqrt(putOptionVariance/paths);

        Realv BSPutPrice = BlackScholesPut(S0, r,  d,  T, sigma,  K);

        std::cout << " Put option estimated price " << putOptionEstimate << " with standard error " << sterr << "\n";
        std::cout << " Put option formula price " << BSPutPrice << "\n";
    }


    Realv Step1Sum=0.0;
    Realv Step1SumSq=0.0;
    Realv Step2Sum=0.0;
    Realv Step2SumSq=0.0;

    Realv prodSum=0.0;


    for (int p=1; p <=paths; ++p)
    {
        Step1Sum += normals_mat(0,p);
        Step1SumSq += normals_mat(0,p)*normals_mat(0,p);
        Step2Sum += normals_mat(1,p);
        Step2SumSq += normals_mat(1,p)*normals_mat(1,p);
        prodSum +=  normals_mat(0,p)*normals_mat(1,p);

    }
    std::cout << "mean normal 1" << Step1Sum/paths << " Mean of square normal 1:" << Step1SumSq/paths << "\n"; 
    std::cout << "mean normal 2" << Step2Sum/paths << " Mean of square normal 2:" << Step2SumSq/paths << "\n"; 
    std::cout << " mean of product of normal 1 and normal 2:" << prodSum/paths << "\n"; 


    char c;
    std::cin >> c;
    return 0;

}


int ComputeGreeks(Realv S0,  const std::vector<Realv>& discontinuityLevels_vec, Realv sigma, Realv r, Realv d,  //int n_poweroftwo,
                  int paths,
                  int n_steps, int normaloffset, Realv dt, Realv drift_per_step, Realv stdev_per_step,
                  int n_vectors, const std::vector<Realv>& uniforms_vec,
                  int maxFlows,
                  const std::vector<Realv>& normals_vec,
                  const std::vector<Realv>& logSpots_vec,
                  const std::vector<Realv>& spots_vec,
                  const std::vector<Realv>& drifts_vec,
                  const std::vector<Realv>& stdev_vec,
                  const std::vector<Realv>& dTs_vec,
                  const std::vector<Realv>& sigmas_vec,
                  const std::vector<Realv>& discountedCashFlows,
                  const std::vector<Realv>& discountedCashFlowStateDerivatives,
                  const std::vector<int>& cashFlowGenerationIndices_vec,
                  bool continuous,
                  MonteCarloStatistics& statisticsObject
                  )
{
    std::vector<Realv> dataPoint_vec(7);

    MatrixConstFacade<Realv> uniforms_mat(&uniforms_vec[0],n_steps,n_vectors);
    MatrixConstFacade<Realv> normals_mat(&normals_vec[0],n_steps,n_vectors);
    MatrixConstFacade<Realv> logSpots_mat(&logSpots_vec[0],paths,n_steps);
    MatrixConstFacade<Realv> spots_mat(&spots_vec[0],paths,n_steps);

    std::vector<Realv> stepSizeSquareRoots(n_steps,sqrt(dt));
    std::vector<Realv> LR_deltas(paths);

    std::vector<int> stepNumbersForVegas(n_steps);
    for (int i=0; i < n_steps; ++i)
        stepNumbersForVegas[i] = i;

    std::vector<Realv> LR_vegas(paths*stepNumbersForVegas.size());

    bool cumulative = false;

    likelihoodRatiosBS_gold (spots_mat,
        normals_mat,
        1,
        stepSizeSquareRoots,
        sigma,
        S0,
        paths,
        n_steps,
        stepNumbersForVegas,
        cumulative,
        LR_deltas,
        LR_vegas
        );

    MatrixConstFacade<Realv> discountedCashFlows_mat(&discountedCashFlows[0],paths,maxFlows);
    MatrixConstFacade<Realv> discountedCashFlowStateDerivatives_mat(&discountedCashFlowStateDerivatives[0],paths,maxFlows);
    MatrixConstFacade<int> cashFlowGenerationIndices_mat(&cashFlowGenerationIndices_vec[0],paths,maxFlows);

    std::vector<int> terminationSteps_vec(paths,n_steps);

    std::vector<Realv> deltas_omegas(paths*n_steps,0.0);
    MatrixFacade<Realv> deltas_omegas_mat(&deltas_omegas[0],paths,n_steps);

    std::vector<Realv> LR_deltas_pad(paths*n_steps,0.0);
    MatrixFacade<Realv>  LR_deltas_mat(&LR_deltas_pad[0],paths,n_steps);
    for (int i=0; i < paths; ++i)
        LR_deltas_mat(i,0) = LR_deltas[i];


    std::vector<Realv> pathwiseGreekEstimates_vec(paths);


    mixedLRPathwiseDeltas_gold(discountedCashFlows_mat,
        discountedCashFlowStateDerivatives_mat,
        cashFlowGenerationIndices_mat,
        terminationSteps_vec,
        paths, 
        deltas_omegas_mat,
        LR_deltas_mat,
        pathwiseGreekEstimates_vec 
        );


    //   Realv deltaVariance;
    //    MonteCarloSimpleStatistics(pathwiseGreekEstimates_vec.begin(),pathwiseGreekEstimates_vec.end(),delta1Estimate, deltaVariance, delta1Sterr);



    // now pure LR vega

    std::vector<Realv> pathLRDeltaEstimates2_vec(paths);
    std::vector<Realv> pathVegaEstimates_vec(paths);


    MatrixFacade<Realv> vegas_w_mat(&LR_vegas[0],paths,n_steps);

    std::vector<Realv> vegas_omegas_vec(paths*n_steps,0.0);

    MatrixFacade<Realv> vegas_omegas_mat(&vegas_omegas_vec[0],paths,n_steps);


    mixedLRPathwiseDeltasVega_gold( discountedCashFlows_mat,
        discountedCashFlowStateDerivatives_mat,
        cashFlowGenerationIndices_mat,
        terminationSteps_vec,
        n_steps,
        paths, 
        deltas_omegas_mat,
        LR_deltas_mat,
        vegas_omegas_mat,
        vegas_w_mat,                                  
        pathLRDeltaEstimates2_vec , 
        pathVegaEstimates_vec  
        );
    /*
    Realv  varianceVega;
    MonteCarloSimpleStatistics(pathVegaEstimates_vec.begin(),pathVegaEstimates_vec.begin()+paths,vegaLREstimate,varianceVega, vegaLRSterr);

    Realv  varianceDelta;
    MonteCarloSimpleStatistics(pathLRDeltaEstimates2_vec.begin(), pathLRDeltaEstimates2_vec.begin()+paths,deltaLREstimate,varianceDelta, deltaLRSterr);
    */

    std::vector<Realv> pathwiseDeltaEstimates_vec(paths,0.0);
    std::vector<Realv> pathwiseVegaEstimates_vec(paths,0.0);


    if (continuous)
    {
        // now pure pathwise methods

        std::vector<Realv> vegas_w_path_vec(paths*n_steps);
        std::vector<Realv> vegas_omega_path_vec(paths*n_steps);
        std::vector<Realv> deltas_w_path_vec(paths*n_steps);
        std::vector<Realv> deltas_omega_path_vec(paths*n_steps);


        MatrixFacade<Realv> vegaspath_w_mat(&vegas_w_path_vec[0],paths,n_steps);
        MatrixFacade<Realv> vegaspath_omegas_mat(&vegas_omega_path_vec[0],paths,n_steps);
        MatrixFacade<Realv> deltaspath_w_mat(&deltas_w_path_vec[0],paths,n_steps);
        MatrixFacade<Realv> deltaspath_omegas_mat(&deltas_omega_path_vec[0],paths,n_steps);

        pathwiseBSweights_gold(spots_mat,
            normals_mat,
            1,
            stepSizeSquareRoots,
            sigma,
            S0,
            paths,
            n_steps,
            deltaspath_w_mat,
            vegaspath_w_mat,
            deltaspath_omegas_mat,
            vegaspath_omegas_mat
            );



        mixedLRPathwiseDeltasVega_gold( discountedCashFlows_mat,
            discountedCashFlowStateDerivatives_mat,
            cashFlowGenerationIndices_mat,
            terminationSteps_vec,
            n_steps,
            paths, 
            deltaspath_omegas_mat,
            deltaspath_w_mat,
            vegaspath_omegas_mat,
            vegaspath_w_mat,                                  
            pathwiseDeltaEstimates_vec , 
            pathwiseVegaEstimates_vec  
            );
        /*
        Realv  variancePathVega;
        MonteCarloSimpleStatistics(pathwiseVegaEstimates_vec.begin(),pathwiseVegaEstimates_vec.begin()+paths,vegaPWEstimate,variancePathVega, vegaPWSterr);
        Realv variancePathDelta;
        MonteCarloSimpleStatistics(pathwiseDeltaEstimates_vec.begin(), pathwiseDeltaEstimates_vec.begin()+paths,deltaPWEstimate,variancePathDelta, deltaPWSterr);
        */
    }


    std::vector<Realv> oppwiseDeltaEstimates_vec(paths);
    std::vector<Realv> oppwiseVegaEstimates_vec(paths);

    // opp method

    {
        std::vector<Realv> vegas_w_opp_vec(paths*n_steps);
        std::vector<Realv> vegas_omega_opp_vec(paths*n_steps);
        std::vector<Realv> deltas_w_opp_vec(paths*n_steps);
        std::vector<Realv> deltas_omega_opp_vec(paths*n_steps);


        MatrixFacade<Realv> vegasopp_w_mat(&vegas_w_opp_vec[0],paths,n_steps);
        MatrixFacade<Realv> vegasopp_omegas_mat(&vegas_omega_opp_vec[0],paths,n_steps);
        MatrixFacade<Realv> deltasopp_w_mat(&deltas_w_opp_vec[0],paths,n_steps);
        MatrixFacade<Realv> deltasopp_omegas_mat(&deltas_omega_opp_vec[0],paths,n_steps);

        std::vector<Realv> stdevs_vec(n_steps,stdev_per_step);
        std::vector<Realv> logDrifts_vec(n_steps,drift_per_step);

        std::vector<Realv> discontinuityLogLevels_vec(discontinuityLevels_vec.size());

        for (unsigned t=0; t < discontinuityLogLevels_vec.size(); ++t)
            discontinuityLogLevels_vec[t] = log(discontinuityLevels_vec[t]);

        oppDeltasVegasBS_gold(logSpots_mat,
            spots_mat,
            uniforms_mat, // the uniforms used to produced the paths
            normals_mat, // the Gaussians used to produced the paths
            normaloffset,
            sigmas_vec,
            dTs_vec,
            stepSizeSquareRoots,     
            logDrifts_vec,
            stdevs_vec,          
            log(S0),
            S0,
            discontinuityLogLevels_vec, // must be increasing
            paths,
            n_steps,
            deltasopp_omegas_mat,
            deltasopp_w_mat,
            vegasopp_omegas_mat,
            vegasopp_w_mat
            );




        mixedLRPathwiseDeltasVega_gold( discountedCashFlows_mat,
            discountedCashFlowStateDerivatives_mat,
            cashFlowGenerationIndices_mat,
            terminationSteps_vec,
            n_steps,
            paths, 
            deltasopp_omegas_mat,
            deltasopp_w_mat,
            vegasopp_omegas_mat,
            vegasopp_w_mat,                                  
            oppwiseDeltaEstimates_vec , 
            oppwiseVegaEstimates_vec  
            );
        /*
        Realv  variancePathVega;
        MonteCarloSimpleStatistics(oppwiseVegaEstimates_vec.begin(),oppwiseVegaEstimates_vec.begin()+paths,vegaOppEstimate,variancePathVega, vegaOppSterr);
        Realv variancePathDelta;
        MonteCarloSimpleStatistics(oppwiseDeltaEstimates_vec.begin(), oppwiseDeltaEstimates_vec.begin()+paths,deltaOppEstimate,variancePathDelta, deltaOppSterr);
        */
    }   
    for (int p=0; p < paths; ++p)
    {
        dataPoint_vec[0] =pathwiseGreekEstimates_vec[p]; // actually LR deltas
        dataPoint_vec[1] =pathLRDeltaEstimates2_vec[p];
        dataPoint_vec[2] =pathVegaEstimates_vec[p];
        dataPoint_vec[3] =pathwiseDeltaEstimates_vec[p];
        dataPoint_vec[4] =pathwiseVegaEstimates_vec[p];
        dataPoint_vec[5] =oppwiseDeltaEstimates_vec[p];
        dataPoint_vec[6] =oppwiseVegaEstimates_vec[p];

        statisticsObject.AddDataVector(dataPoint_vec);



    }

    return 0;

}


void OneDBSMonteCarloGreekCalls(int paths, int verbose, int dataset, int numberFlows, int power, Realv K, MonteCarloStatistics& statisticsObject)
{

    Realv S0 ;
    Realv sigma;
    Realv r;
    Realv d ;
    Realv T;
    int n_poweroftwo;
    GetBSParameters(dataset,  S0,  sigma,  r,  d, T,n_poweroftwo);

    Realv formulaDelta=0.0;
    Realv formulaVega=0.0;

    for (int f=0; f < numberFlows; ++f)
    {
        formulaDelta +=  BlackScholesPowerCallDelta( S0, r,  d,  (f+1)*T/numberFlows, sigma, K,power);
        formulaVega +=  BlackScholesPowerCallVega( S0, r,  d,  (f+1)*T/numberFlows, sigma, K,power);
    }

    std::cout << "\nformula delta: "<< formulaDelta << "\n";
    std::cout << "formula vega: "<< formulaVega << "\n\n";

    int n_steps,  normaloffset, n_vectors;
    Realv dt, drift_per_step,  stdev_per_step;
    std::vector<Realv> uniforms_vec,normals_vec,logSpots_vec,spots_vec,drifts_vec, stdev_vec,dTs_vec, sigmas_vec;

    n_steps=  intPower(2,n_poweroftwo);

    GenerateBSPathData(S0,  sigma,  r,  d,  T,  paths,
        n_steps, normaloffset,  dt, drift_per_step,  stdev_per_step,
        n_vectors, uniforms_vec,
        normals_vec,
        logSpots_vec,
        spots_vec,
        drifts_vec,
        stdev_vec,
        dTs_vec,
        sigmas_vec);

    std::vector<Realv> Ls(1);
    Ls[0] = K;

    std::vector<Realv> discountedCashFlows;
    std::vector<Realv> discountedCashFlowStateDerivatives;
    std::vector<int> cashFlowGenerationIndices_vec;
    MatrixFacade<Realv> spots_mat(&spots_vec[0],paths,n_steps);

    GenerateCashFlowsPowerCalls( numberFlows,   discountedCashFlows,
        discountedCashFlowStateDerivatives ,
        cashFlowGenerationIndices_vec,
        power,
        paths, 
        K,
        T,
        r,
        n_steps,
        spots_mat
        );

    bool continuous = (power >0);

    ComputeGreeks(S0,   Ls,  sigma, r, d,     paths,
        n_steps,  normaloffset,  dt,  drift_per_step,  stdev_per_step,
        n_vectors,  uniforms_vec,
        numberFlows,
        normals_vec,
        logSpots_vec,
        spots_vec,
        drifts_vec,
        stdev_vec,
        dTs_vec,
        sigmas_vec,
        discountedCashFlows,
        discountedCashFlowStateDerivatives,
        cashFlowGenerationIndices_vec,
        continuous,
        statisticsObject);




}


void OneDBSMonteCarloGreekDoubleDigitals(int paths, int verbose, int dataset, int numberFlows,  Realv K1, Realv K2, 
                                         MonteCarloStatistics& statisticsObject)
{

    Realv S0 ;

    Realv sigma;
    Realv r;
    Realv d ;
    Realv T;
    int n_poweroftwo;
    GetBSParameters(dataset,  S0,   sigma,  r,  d, T,n_poweroftwo);

    Realv formulaDelta=0.0;
    Realv formulaVega=0.0;

    for (int f=0; f < numberFlows; ++f)
    {
        formulaDelta +=  BlackScholesDoubleDigitalDelta( S0, r,  d,  (f+1)*T/numberFlows, sigma, K1,K2);
        formulaVega +=  BlackScholesDoubleDigitalVega( S0, r,  d,  (f+1)*T/numberFlows, sigma, K1,K2);
    }

    std::cout << "\nformula delta: "<< formulaDelta << "\n";
    std::cout << "formula vega: "<< formulaVega << "\n\n";

    int n_steps,  normaloffset, n_vectors;
    Realv dt, drift_per_step,  stdev_per_step;
    std::vector<Realv> uniforms_vec,normals_vec,logSpots_vec,spots_vec,drifts_vec, stdev_vec,dTs_vec, sigmas_vec;

    n_steps = intPower(2,n_poweroftwo);

    GenerateBSPathData(S0,   sigma,  r,  d,  T, paths,
        n_steps, normaloffset,  dt, drift_per_step,  stdev_per_step,
        n_vectors, uniforms_vec,
        normals_vec,
        logSpots_vec,
        spots_vec,
        drifts_vec,
        stdev_vec,
        dTs_vec,
        sigmas_vec);

    std::vector<Realv> Ls(2);
    Ls[0] = K1;
    Ls[1] = K2;

    std::vector<Realv> discountedCashFlows;
    std::vector<Realv> discountedCashFlowStateDerivatives;
    std::vector<int> cashFlowGenerationIndices_vec;
    MatrixFacade<Realv> spots_mat(&spots_vec[0],paths,n_steps);

    GenerateCashFlowsDoubleDigitals( numberFlows,   discountedCashFlows,
        discountedCashFlowStateDerivatives ,
        cashFlowGenerationIndices_vec,
        paths, 
        K1,K2,
        T,
        r,
        n_steps,
        spots_mat
        );

    bool continuous = false;


    ComputeGreeks(S0,   Ls,  sigma, r, d,    paths,
        n_steps,  normaloffset,  dt,  drift_per_step,  stdev_per_step,
        n_vectors,  uniforms_vec,
        numberFlows,
        normals_vec,
        logSpots_vec,
        spots_vec,
        drifts_vec,
        stdev_vec,
        dTs_vec,
        sigmas_vec,
        discountedCashFlows,
        discountedCashFlowStateDerivatives,
        cashFlowGenerationIndices_vec,
        continuous,
        statisticsObject);



}


void OneDBSMonteCarloGreekTriggerableRangeAccrual(int pathsPerBatch,
                                                  int batches,
                                                  bool verbose, 
                                                  int dataset, 
                                                  int periods,
                                                  int daysPerPeriod,
                                                  Realv accrualLevel, 
                                                  Realv triggerBarrier,
                                                  int daysForTriggering,
                                                  Realv rebate,
                                                  Realv couponRate,
                                                  Realv callNotional,
                                                  Realv callStrike,
                                                  MonteCarloStatistics& statisticsObject)
{


    int n_steps = daysPerPeriod * periods;

    std::vector<unsigned int> scramblers_vec(n_steps);
    MersenneTwisterUniformRng rng;

    Realv S0 ;

    Realv sigma;
    Realv r;
    Realv d ;
    Realv T;

    int n_poweroftwo;
    GetBSParameters(dataset,  S0,   sigma,  r,  d, T,n_poweroftwo);
    std::vector<Realv> Ls(2);
    Ls[0] = accrualLevel;
    Ls[1] = triggerBarrier;



    for (int batch=0; batch<batches; ++batch)
    {
        rng.getInts(scramblers_vec.begin(),scramblers_vec.end());



        int  normaloffset, n_vectors;
        Realv dt, drift_per_step,  stdev_per_step;
        std::vector<Realv> uniforms_vec,normals_vec,logSpots_vec,spots_vec,drifts_vec, stdev_vec,dTs_vec, sigmas_vec;

        GetSobolUniformsWithScrambling( uniforms_vec,                  
            normaloffset,
            n_vectors,
            pathsPerBatch, 
            n_steps,
            scramblers_vec);

        normals_vec.resize(uniforms_vec.size());
        inverseCumulativeGold<Realv> inv;
        std::transform(uniforms_vec.begin(), uniforms_vec.end(),normals_vec.begin(),inv);

        GenerateBSPathDataGivenVariates(S0,   sigma,  r,  d,  T,  pathsPerBatch,
            n_steps, normaloffset,  dt, drift_per_step,  stdev_per_step,
            n_vectors, uniforms_vec,
            normals_vec,
            logSpots_vec,
            spots_vec,
            drifts_vec,
            stdev_vec,
            dTs_vec,
            sigmas_vec);


        std::vector<Realv> discountedCashFlows;
        std::vector<Realv> discountedCashFlowStateDerivatives;
        std::vector<int> cashFlowGenerationIndices_vec;
        MatrixConstFacade<Realv> spots_mat(&spots_vec[0],pathsPerBatch,n_steps);

        GenerateCashFlowsTriggerableRangeAccrual( discountedCashFlows,
            discountedCashFlowStateDerivatives ,
            cashFlowGenerationIndices_vec,
            pathsPerBatch,  
            accrualLevel,
            triggerBarrier,
            callNotional,
            callStrike,
            daysForTriggering,
            rebate,
            couponRate,
            daysPerPeriod,
            periods,
            T,
            r,                                         
            spots_mat
            );



        bool continuous = false;



        ComputeGreeks(S0,   Ls,  sigma, r, d,   pathsPerBatch,
            n_steps,  normaloffset,  dt,  drift_per_step,  stdev_per_step,
            n_vectors,  uniforms_vec,
            n_steps,
            normals_vec,
            logSpots_vec,
            spots_vec,
            drifts_vec,
            stdev_vec,
            dTs_vec,
            sigmas_vec,
            discountedCashFlows,
            discountedCashFlowStateDerivatives,
            cashFlowGenerationIndices_vec,
            continuous,
            statisticsObject);




    }




}

void OneDBSMonteCarloGreekCorridor(int pathsPerBatch,
                                   int batches,
                                   bool verbose, 
                                   int dataset, 
                                   int days,
                                   Realv rebate,
                                   Realv LowerBarrier, 
                                   Realv UpperBarrier,
                                   int daysForTriggering,
                                   MonteCarloStatistics& statisticsObject,
                                   bool consecutive)
{


    int n_steps = days;

    std::vector<unsigned int> scramblers_vec(n_steps);
    MersenneTwisterUniformRng rng;

    Realv S0 ;

    Realv sigma;
    Realv r;
    Realv d ;
    Realv T;

    int n_poweroftwo;
    GetBSParameters(dataset,  S0,   sigma,  r,  d, T,n_poweroftwo);
    std::vector<Realv> Ls(2);
    Ls[0] = LowerBarrier;
    Ls[1] = UpperBarrier;



    for (int batch=0; batch<batches; ++batch)
    {
        rng.getInts(scramblers_vec.begin(),scramblers_vec.end());



        int  normaloffset, n_vectors;
        Realv dt, drift_per_step,  stdev_per_step;
        std::vector<Realv> uniforms_vec,normals_vec,logSpots_vec,spots_vec,drifts_vec, stdev_vec,dTs_vec, sigmas_vec;

        GetSobolUniformsWithScrambling( uniforms_vec,                  
            normaloffset,
            n_vectors,
            pathsPerBatch, 
            n_steps,
            scramblers_vec);

        normals_vec.resize(uniforms_vec.size());
        inverseCumulativeGold<Realv> inv;
        std::transform(uniforms_vec.begin(), uniforms_vec.end(),normals_vec.begin(),inv);

        GenerateBSPathDataGivenVariates(S0,   sigma,  r,  d,  T,  pathsPerBatch,
            n_steps, normaloffset,  dt, drift_per_step,  stdev_per_step,
            n_vectors, uniforms_vec,
            normals_vec,
            logSpots_vec,
            spots_vec,
            drifts_vec,
            stdev_vec,
            dTs_vec,
            sigmas_vec);


        std::vector<Realv> discountedCashFlows;
        std::vector<Realv> discountedCashFlowStateDerivatives;
        std::vector<int> cashFlowGenerationIndices_vec;
        MatrixConstFacade<Realv> spots_mat(&spots_vec[0],pathsPerBatch,n_steps);

        GenerateCashFlowsCorridor(  discountedCashFlows,  discountedCashFlowStateDerivatives ,
            cashFlowGenerationIndices_vec,
            pathsPerBatch,  
            LowerBarrier,
            UpperBarrier,
            daysForTriggering,
            rebate,
            days,
            T,
            r,                                         
            spots_mat,
            consecutive
            );



        bool continuous = false;


        int maxFlows =1;

        ComputeGreeks(S0,   Ls,  sigma, r, d,   pathsPerBatch,
            n_steps,  normaloffset,  dt,  drift_per_step,  stdev_per_step,
            n_vectors,  uniforms_vec,
            maxFlows,
            normals_vec,
            logSpots_vec,
            spots_vec,
            drifts_vec,
            stdev_vec,
            dTs_vec,
            sigmas_vec,
            discountedCashFlows,
            discountedCashFlowStateDerivatives,
            cashFlowGenerationIndices_vec,
            continuous,
            statisticsObject);




    }




}
int BSGreeksMain()
{
    int BatchSize = 16383;
    int numberBatches =1;
    std::vector<std::string> names;
    names.push_back("LR Delta1");
    names.push_back("LR Delta2");
    names.push_back("LR Vega");
    names.push_back("PW Delta");
    names.push_back("PW Vega");
    names.push_back("Opp Delta");
    names.push_back("Opp Vega");
    int dataPoints = names.size();



    Realv tick =1.0/CLOCKS_PER_SEC;
    int startTime = clock();

    int periods =1;
    int daysPerPeriod = 360;
    int daysForTriggering =3;
    Realv callNotional =1.0;
    Realv rebate =1.0;
    Realv couponRate = 0.05;

    Realv accrualLevel;
    Realv triggerBarrier;


    int noLowers = 11;
    int noUppers =11;

    std::vector<Realv> res(noLowers*noUppers*2);
    CubeFacade<Realv> res_cube(&res[0],noLowers,noUppers,2);

    //    Realv callStrike = triggerBarrier;
    bool consecutive = false;

  
    for (int dataSet=4; dataSet<5; ++dataSet)
        for (int i=0; i < 11; ++i)
            for (int j=0; j < 11; ++j)
            {
                
                MonteCarloStatisticsBatched statisticsObject(BatchSize, dataPoints, names);
                accrualLevel = 75.0+2*i;
                triggerBarrier = 105.0+2*j;

                Realv LowerBarrier = accrualLevel;

                Realv UpperBarrier = triggerBarrier;


                std::cout << "\n\n\n LowerBarrier, " << accrualLevel << ", UpperBarrier," << triggerBarrier << "\n";
                std::cout << "periods, " << periods << ", days per period, " << daysPerPeriod << "\n";
                std::cout << "daysForTriggering, " << daysForTriggering <<  "\n";


                std::cout << "\n data set " << dataSet << "\n";
                std::cout << "\n consecutive barrier " << consecutive << "\n";

                OneDBSMonteCarloGreekCorridor(BatchSize,numberBatches, false,  
                    dataSet, 
                    daysPerPeriod,
                    rebate,
                    LowerBarrier, 
                    UpperBarrier,
                    daysForTriggering,
                    statisticsObject,
                    consecutive);
               

                std::vector<std::vector<Realv> > output(statisticsObject.GetStatistics());
                std::vector<std::string> names(statisticsObject.GetStatisticsNames());

                std::cout << "\n\n";

                for (size_t m=0; m < names.size(); ++m)
                    std::cout << names[m] << ",";

                for (size_t l=0; l < output.size(); ++l)
                {
                    std::cout << "\n";
                    for (size_t m=0; m < output[l].size(); ++m)
                        std::cout << output[l][m] << ", ";
                }

                res_cube(i,j,0) = output[2][2];
                res_cube(i,j,1) = output[2][6];

            }

            int endTime = clock();

            std::cout << " \n Time taken, " << (endTime-startTime)*tick << "\n";

            for (int i=0; i < noLowers; ++i)
                for (int j=0; j < noUppers; ++j)
                {
                    std::cout << i << ", " << j << "," << res_cube(i,j,0) << "," << res_cube(i,j,1) << "\n";
                }

            // OneDBSMonteCarloPrice(16383, false);
            //  OneDBSMonteCarloGreekDoubleDigitals(16383, false,0 , 2,  99, 105);

            //  OneDBSMonteCarloGreekCalls(16383, false,1,2,1,100);
            //  OneDBSMonteCarloGreekCalls(16383, false,1,2,0,95);

            //  char c;
            //   std::cin >> c;

return 0;
}
